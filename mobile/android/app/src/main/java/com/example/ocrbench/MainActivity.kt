package com.example.ocrbench

import android.app.Activity
import android.os.Bundle
import android.os.Debug
import android.util.Log
import android.widget.TextView
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

class MainActivity : Activity() {
    private val TAG = "OCRBench"
    private lateinit var textView: TextView
    private val results = mutableListOf<String>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        textView = TextView(this).apply { textSize = 14f }
        setContentView(textView)

        Thread {
            runBenchmark()
        }.start()
    }

    private fun runBenchmark() {
        val env = OrtEnvironment.getEnvironment()
        val nativeHeapBefore = Debug.getNativeHeapAllocatedSize()

        // モデルリスト
        val models = listOf(
            "PP-OCRv6_small_det_int8.ort" to "det",
            "PP-OCRv6_small_rec_int8.ort" to "rec",
            "PP-LCNet_x1_0_doc_ori_int8.ort" to "doc_ori",
            "PP-LCNet_x1_0_textline_ori_int8.ort" to "textline_ori",
            "model_int8_emb8.ort" to "labeler"
        )

        val sessions = mutableMapOf<String, OrtSession>()
        for ((filename, name) in models) {
            try {
                val buffer = assets.open(filename).readBytes()
                val session = env.createSession(buffer)
                sessions[name] = session
                log("Loaded: $name (${buffer.size / 1024 / 1024}MB)")
            } catch (e: Exception) {
                log("Failed to load $name: ${e.message}")
            }
        }

        // テストデータ読み込み
        val testPrefixes = listOf("receipt", "receipt9")
        val nIterations = 10

        for (prefix in testPrefixes) {
            log("\n=== $prefix ===")

            // doc_ori
            loadAndRun(env, sessions["doc_ori"], "${prefix}_doc_ori", nIterations, "doc_ori")

            // det
            loadAndRun(env, sessions["det"], "${prefix}_det", nIterations, "det")

            // rec (cropごと)
            var i = 0
            while (true) {
                val name = "${prefix}_rec_$i"
                if (!assetExists(name)) break
                loadAndRun(env, sessions["rec"], name, nIterations, "rec")
                i++
            }
            log("  rec crops: $i")

            // textline_ori
            loadAndRun(env, sessions["textline_ori"], "${prefix}_textline", nIterations, "textline_ori")

            // labeler
            loadLabelerAndRun(env, sessions["labeler"], prefix, nIterations)
        }

        // メモリ使用量
        val nativeHeapAfter = Debug.getNativeHeapAllocatedSize()
        log("\n=== Memory ===")
        log("Native heap: ${(nativeHeapAfter - nativeHeapBefore) / 1024 / 1024}MB")

        // セッション開放
        sessions.values.forEach { it.close() }
    }

    private fun loadAndRun(env: OrtEnvironment, session: OrtSession?,
                           tensorName: String, nIterations: Int, modelName: String) {
        session ?: return
        try {
            // npyファイルからデータを読み込み(簡易実装: テキストファイルから)
            val tensorData = loadTensorFromAssets(tensorName) ?: return
            val inputName = session.inputNames.first()

            // ウォームアップ
            val output = session.run(mapOf(inputName to tensorData))

            // 計測
            val times = mutableListOf<Long>()
            for (i in 0 until nIterations) {
                val start = System.nanoTime()
                session.run(mapOf(inputName to tensorData))
                val elapsed = System.nanoTime() - start
                times.add(elapsed)
            }

            val avgMs = times.average() / 1_000_000
            val maxMs = times.max().toDouble() / 1_000_000
            log("  $modelName: avg=${String.format("%.1f", avgMs)}ms max=${String.format("%.1f", maxMs)}ms")
        } catch (e: Throwable) {
            Log.e(TAG, "  $modelName failed", e)
            log("  $modelName: error - ${e.javaClass.simpleName}: ${e.message}")
        }
    }

    private fun loadLabelerAndRun(env: OrtEnvironment, session: OrtSession?,
                                  prefix: String, nIterations: Int) {
        session ?: return
        try {
            val idsData = loadTensorFromAssets("${prefix}_labeler_ids") ?: return
            val maskData = loadTensorFromAssets("${prefix}_labeler_mask") ?: return
            val inputIdsName = session.inputNames.first { it.contains("input_ids") }
            val attentionMaskName = session.inputNames.first { it.contains("attention_mask") }

            val feeds = mapOf(inputIdsName to idsData, attentionMaskName to maskData)

            // ウォームアップ
            session.run(feeds)

            // 計測
            val times = mutableListOf<Long>()
            for (i in 0 until nIterations) {
                val start = System.nanoTime()
                session.run(feeds)
                val elapsed = System.nanoTime() - start
                times.add(elapsed)
            }

            val avgMs = times.average() / 1_000_000
            val maxMs = times.max().toDouble() / 1_000_000
            log("  labeler: avg=${String.format("%.1f", avgMs)}ms max=${String.format("%.1f", maxMs)}ms")
        } catch (e: Throwable) {
            Log.e(TAG, "  labeler failed", e)
            log("  labeler: error - ${e.javaClass.simpleName}: ${e.message}")
        }
    }

    private fun loadTensorFromAssets(name: String): OnnxTensor? {
        return try {
            // メタデータ(shape/dtype)を読む。データ本体はリトルエンディアンの生バイト列(.bin)
            val metaJson = assets.open("$name.json").bufferedReader().use { it.readText() }
            val shapeMatch = Regex("\"shape\":\\s*\\[(.+?)\\]", RegexOption.DOT_MATCHES_ALL).find(metaJson)
            val dtypeMatch = Regex("\"dtype\":\\s*\"(.+?)\"").find(metaJson)
            if (shapeMatch == null || dtypeMatch == null) return null

            val shape = shapeMatch.groupValues[1].split(",").map { it.trim().toLong() }.toLongArray()
            val dtype = dtypeMatch.groupValues[1]
            val expected = shape.fold(1L) { a, b -> a * b }

            val bytes = assets.open("$name.bin").readBytes()
            val env = OrtEnvironment.getEnvironment()
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

            when {
                dtype.contains("float32") -> {
                    require(bytes.size.toLong() == expected * 4) { "size mismatch: $name" }
                    val fb = FloatBuffer.allocate(bytes.size / 4)
                    fb.put(buf.asFloatBuffer()); fb.rewind()
                    OnnxTensor.createTensor(env, fb, shape)
                }
                dtype.contains("int64") -> {
                    require(bytes.size.toLong() == expected * 8) { "size mismatch: $name" }
                    val lb = LongBuffer.allocate(bytes.size / 8)
                    lb.put(buf.asLongBuffer()); lb.rewind()
                    OnnxTensor.createTensor(env, lb, shape)
                }
                else -> null
            }
        } catch (e: Exception) {
            log("Failed to load tensor $name: ${e.message}")
            null
        }
    }

    private fun assetExists(name: String): Boolean {
        return try {
            assets.open("$name.bin").close()
            true
        } catch (e: Exception) {
            false
        }
    }

    private fun log(msg: String) {
        Log.i(TAG, msg)
        // resultsへの追記(バックグラウンドスレッド)とjoinToString(UIスレッド)が競合し
        // ConcurrentModificationExceptionでクラッシュしていたため、追記と文字列化を
        // 同じ同期ブロック内で完結させ、UIスレッドには完成済み文字列だけ渡す
        val snapshot: String
        synchronized(results) {
            results.add(msg)
            snapshot = results.joinToString("\n")
        }
        runOnUiThread {
            textView.text = snapshot
        }
    }
}
