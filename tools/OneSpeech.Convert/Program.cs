using System.Security.Cryptography;
using OneSpeech.Catalog.Conversion;
using OneSpeech.Formats;
using OneSpeech.Whisper;
if (args is ["int8-whisper-batch", var ibModel, var ibSplit, var ibOutput, var ibBegin, var ibCount]) { Int8Evaluation.Batch(ibModel, ibSplit, ibOutput, int.Parse(ibBegin, System.Globalization.CultureInfo.InvariantCulture), int.Parse(ibCount, System.Globalization.CultureInfo.InvariantCulture)); return 0; }
if (args is ["int8-whisper-metrics", var imModel, var imClip, var imOutput, var imThreads]) { Int8Evaluation.Metrics(imModel, imClip, imOutput, int.Parse(imThreads, System.Globalization.CultureInfo.InvariantCulture)); return 0; }
if (args is ["int8-candidate", var candidateInput, var candidateOutput]) { Int8Conversion.Run(candidateInput, candidateOutput, verifyConverted: false); return 0; }
if (args is ["int8-convert", var int8Input, var int8Output]) { Int8Conversion.Run(int8Input, int8Output); return 0; }
if (args is ["speaker-calibrate", var calibrationTrials, var calibrationEmbeddings]) { SpeakerCalibration.Run(calibrationTrials, calibrationEmbeddings); return 0; }
if (args is ["nemotron-asr-live-batch", var lbDir, var lbCorpus, var lbOut, var lbChunk, var lbBegin, var lbCount]) { NemotronAsrEvaluation.Batch(lbDir, lbCorpus, lbOut, int.Parse(lbChunk, System.Globalization.CultureInfo.InvariantCulture), int.Parse(lbBegin, System.Globalization.CultureInfo.InvariantCulture), int.Parse(lbCount, System.Globalization.CultureInfo.InvariantCulture), publishedEvents: true); return 0; }
if (args is ["nemotron-asr-latency-truth", var ltDir, var ltBase, var ltHyp, var ltTruth]) { NemotronAsrLatency.Run(ltDir, ltBase, ltHyp, ltTruth); return 0; }
if (args is ["nemotron-asr-latency", var nlDir, var nlBase, var nlHyp]) { NemotronAsrLatency.Run(nlDir, nlBase, nlHyp); return 0; }
if (args is ["nemotron-asr-eval-slice", var nsCorpus, var nsOurs, var nsRef, var nsOut, var nsCount]) { NemotronAsrChecks.Prepare(nsCorpus, nsOurs, nsRef, nsOut, int.Parse(nsCount, System.Globalization.CultureInfo.InvariantCulture)); return 0; }
if (args is ["nemotron-asr-metrics", var ntDir, var ntClip, var ntOut, var ntChunk, var ntThreads]) { NemotronAsrMetrics.Run(ntDir, ntClip, ntOut, int.Parse(ntChunk, System.Globalization.CultureInfo.InvariantCulture), int.Parse(ntThreads, System.Globalization.CultureInfo.InvariantCulture)); return 0; }
if (args is ["nemotron-asr-metrics", var nmDir, var nmClip, var nmOut, var nmChunk]) { NemotronAsrMetrics.Run(nmDir, nmClip, nmOut, int.Parse(nmChunk, System.Globalization.CultureInfo.InvariantCulture)); return 0; }
if (args is ["nemotron-asr-catalog", var ncPub, var ncConv, var ncCat, var ncLicense, var ncOut]) { NemotronAsrChecks.Catalog(ncPub, ncConv, ncCat, ncLicense, ncOut); return 0; }
if (args is ["nemotron-asr-vocabulary", var nvDir, var nvFixture]) { NemotronAsrChecks.Vocabulary(nvDir, nvFixture); return 0; }
if (args is ["nemotron-asr-eval", var neCorpus, var neOurs, var neRef, var neOut]) { NemotronAsrChecks.Prepare(neCorpus, neOurs, neRef, neOut); return 0; }
if (args is ["nemotron-asr-batch", var nbDir, var nbCorpus, var nbOut, var nbChunk, var nbBegin, var nbCount]) { NemotronAsrEvaluation.Batch(nbDir, nbCorpus, nbOut, int.Parse(nbChunk, System.Globalization.CultureInfo.InvariantCulture), int.Parse(nbBegin, System.Globalization.CultureInfo.InvariantCulture), int.Parse(nbCount, System.Globalization.CultureInfo.InvariantCulture)); return 0; }
if (args is ["nemotron-asr-probe-caches", var cacheModel, var cacheProbe]) { NemotronAsrTools.ProbeCaches(cacheModel, cacheProbe); return 0; }
if (args is ["nemotron-asr-probe-decoder", var decoderModel, var decoderNative, var decoderHf]) { NemotronAsrTools.ProbeDecoder(decoderModel, decoderNative, decoderHf); return 0; }
if (args is ["nemotron-asr-convert-streaming", var asrNemotronStreamInput, var asrNemotronStreamOutput]) { NemotronAsrConverter.Convert(asrNemotronStreamInput, asrNemotronStreamOutput, CancellationToken.None, streaming: true); return 0; }
if (args is ["nemotron-asr-convert-probe", var asrNemotronProbeInput, var asrNemotronProbeOutput]) { NemotronAsrConverter.Convert(asrNemotronProbeInput, asrNemotronProbeOutput, CancellationToken.None, probeOutputs: true); return 0; }
if (args is ["nemotron-asr-convert", var asrNemotronInput, var asrNemotronOutput]) { NemotronAsrConverter.Convert(asrNemotronInput, asrNemotronOutput, CancellationToken.None); return 0; }
if (args is ["nemotron-asr-compare-encoder", var compareModel, var nativeProbe, var hfProbe]) { NemotronAsrTools.ProbeEncoder(compareModel, nativeProbe, transformerProbes: hfProbe); return 0; }
if (args is ["nemotron-asr-probe-encoder", var nemotronModel, var nemotronProbe]) { NemotronAsrTools.ProbeEncoder(nemotronModel, nemotronProbe); return 0; }
if (args is ["nemotron-topk-probe", var nemotronTopKDirectory]) { NemotronTools.TopKProbe(nemotronTopKDirectory); return 0; }
if (args is ["nemotron-score", var nemotronScoreTruth, var nemotronScoreHypothesis, var nemotronScoreReference]) { NemotronBatch.ScoreOnly(nemotronScoreTruth, nemotronScoreHypothesis, nemotronScoreReference); return 0; }
if (args is ["nemotron-cache-map", var nemotronMapNative, var nemotronMapReference, var nemotronMapSilence]) { NemotronTools.CacheMap(nemotronMapNative, nemotronMapReference, nemotronMapSilence); return 0; }
if (args is ["nemotron-stream-probe", var nemotronStreamModel, var nemotronStreamAudio, var nemotronStreamOutput, var nemotronStreamRef]) { NemotronTools.StreamProbe(nemotronStreamModel, nemotronStreamAudio, nemotronStreamOutput, nemotronStreamRef); return 0; }
if (args is ["nemotron-native", var nemotronDirectory, var nemotronRecording, var nemotronResults, var nemotronTruth, var nemotronReference]) { NemotronBatch.Run(nemotronDirectory, nemotronRecording, nemotronResults, nemotronTruth, nemotronReference); return 0; }
if (args is ["nemotron-cache", var nemotronCacheDir]) { NemotronTools.CacheProbe(nemotronCacheDir); return 0; }
if (args is ["nemotron-features", var nemotronAudio, var nemotronMel]) { NemotronTools.Features(nemotronAudio, nemotronMel); return 0; }
if (args is ["nemotron-convert", var nemotronInput, var nemotronOutput]) { NemotronTools.Convert(nemotronInput, nemotronOutput); return 0; }
if (args is ["nemotron-probe", var nemotronGraph, var nemotronFeatures, var nemotronProbeOutput]) { NemotronTools.Probe(nemotronGraph, nemotronFeatures, nemotronProbeOutput); return 0; }
if (args is ["nemotron-compare", var nemotronFirst, var nemotronSecond]) { NemotronTools.Compare(nemotronFirst, nemotronSecond); return 0; }
if (args is ["custom-fixture", var customFixture]) { CustomGraphProbe.Fixture(customFixture); return 0; }
if (args is ["custom-probe", var customKind, var customDirectory, var customCorpus, var customOutput]) { CustomGraphProbe.Run(customKind, customDirectory, customCorpus, customOutput); return 0; }
if (args is ["parakeet-vocabulary", var parakeetTokenizerDirectory, var parakeetTokenizerFixtures]) { ParakeetTools.Vocabulary(parakeetTokenizerDirectory, parakeetTokenizerFixtures); return 0; }
if (args is ["parakeet-eval", var parakeetEvalCorpus, var parakeetOurs, var parakeetReference, var parakeetEvalOutput]) { ParakeetEvaluation.Prepare(parakeetEvalCorpus, parakeetOurs, parakeetReference, parakeetEvalOutput); return 0; }
if (args is ["parakeet-timing", var parakeetTimingReference, var parakeetTimingHypothesis]) { ParakeetEvaluation.Timing(parakeetTimingReference, parakeetTimingHypothesis); return 0; }
if (args is ["parakeet-batch", var parakeetDirectory, var parakeetCorpus, var parakeetResults, var parakeetBegin, var parakeetCount]) { ParakeetTools.Batch(parakeetDirectory, parakeetCorpus, parakeetResults, int.Parse(parakeetBegin, System.Globalization.CultureInfo.InvariantCulture), int.Parse(parakeetCount, System.Globalization.CultureInfo.InvariantCulture)); return 0; }
if (args is ["parakeet-convert", var parakeetInput, var parakeetOutput]) { ParakeetConverter.Convert(parakeetInput, parakeetOutput, CancellationToken.None); return 0; }
if (args is ["parakeet-probe", var parakeetModel, var parakeetProbe]) { ParakeetTools.Probe(parakeetModel, parakeetProbe); return 0; }
if (args is ["moonshine-live-compare", var liveBefore, var liveAfter])
{
    MoonshineLiveProbe.Compare(liveBefore, liveAfter);
    return 0;
}
if (args is ["silero-operators"])
{
    MoonshineAllocationProfile.SileroOperators();
    return 0;
}
if (args is ["moonshine-live-inspect", var inspectDirectory, var inspectOutput])
{
    MoonshineLiveMetrics.Inspect(inspectDirectory, inspectOutput);
    return 0;
}
if (args is ["moonshine-cache-convert", var cacheSource])
{
    _ = OneSpeech.Moonshine.MoonshineProvenance.Verify(cacheSource, converted: false, CancellationToken.None);
    var cc = OneSpeech.Moonshine.MoonshineConfiguration.Load(cacheSource);
    using var ci = File.OpenRead(Path.Combine(cacheSource, "model.safetensors")); using var cw = new SafeTensorFile(ci);
    var cp = new MoonshineGraphWriter(cw); cp.DecoderProject(cc); cp.Save(Path.Combine(cacheSource, "decoder-project.onemodel"), CancellationToken.None);
    var cs = new MoonshineGraphWriter(cw); cs.DecoderStep(cc); cs.Save(Path.Combine(cacheSource, "decoder-step.onemodel"), CancellationToken.None);
    return 0;
}
if (args is ["moonshine-live-metrics", var metricsModel, var metricsClip, var metricsOutput])
{
    MoonshineLiveMetrics.Run(metricsModel, metricsClip, metricsOutput);
    return 0;
}
if (args is ["qwen-precision-repro", var precisionRepro]) { QwenPrecisionProbe.Run(precisionRepro); return 0; }
if (args is ["qwen-cache-paths", var cachePathsRoot]) { QwenTools.CachePaths(cachePathsRoot); return 0; }
if (args is ["qwen-e2e", var e2eCache, var e2eAudio, var e2eJson]) { QwenTools.EndToEnd(e2eCache, e2eAudio, e2eJson); return 0; }
if (args is ["qwen-audio", var canonicalInput, var canonicalOutput]) { QwenTools.CanonicalAudio(canonicalInput, canonicalOutput); return 0; }
if (args is ["qwen-long", var longModel, var longAudio, var longJson, var longPrecision]) { QwenTools.LongForm(longModel, longAudio, longJson, longPrecision == "bf16"); return 0; }
if (args is ["qwen-rebuild", var rebuildModel, var rebuildPrecision])
{
    QwenTools.Rebuild(rebuildModel, rebuildPrecision == "bf16");
    return 0;
}
if (args is ["qwen-reproduce", var reproductionModel, var reproductionPrecision])
{
    QwenTools.Reproduce(reproductionModel, reproductionPrecision == "bf16");
    return 0;
}
if (args is ["qwen-batch-bf16", var bfBatchModel, var bfBatchRoot, var bfBatchSplit, var bfBatchOutput, var bfBatchBegin, var bfBatchCount])
{
    QwenTools.Batch(bfBatchModel, bfBatchRoot, bfBatchSplit, bfBatchOutput, int.Parse(bfBatchBegin, System.Globalization.CultureInfo.InvariantCulture), int.Parse(bfBatchCount, System.Globalization.CultureInfo.InvariantCulture), OneSpeech.Qwen3.Qwen3Precision.BFloat16);
    return 0;
}
if (args is ["qwen-probe-decode-bf16", var bfProbeModel, var bfProbeData])
{
    QwenTools.ProbeDecoder(bfProbeModel, bfProbeData, true);
    return 0;
}
if (args is ["qwen-convert-bf16", var bf16Input, var bf16Output])
{
    OneSpeech.Catalog.Conversion.QwenConverter.Convert(bf16Input, bf16Output, CancellationToken.None, OneSpeech.Qwen3.Qwen3Precision.BFloat16);
    return 0;
}
if (args is ["qwen-convert", var convertQwenInput, var convertQwenOutput])
{
    OneSpeech.Catalog.Conversion.QwenConverter.Convert(convertQwenInput, convertQwenOutput, CancellationToken.None);
    return 0;
}
if (args is ["qwen-catalog", var catalogModel, var catalogJson])
{
    QwenTools.Catalog(catalogModel, catalogJson);
    return 0;
}
if (args is ["qwen-align-batch", var alignmentModel, var alignmentRoot, var alignmentSplit, var alignmentOutput, var alignmentBegin, var alignmentCount])
{
    QwenTools.AlignBatch(alignmentModel, alignmentRoot, alignmentSplit, alignmentOutput, int.Parse(alignmentBegin, System.Globalization.CultureInfo.InvariantCulture), int.Parse(alignmentCount, System.Globalization.CultureInfo.InvariantCulture));
    return 0;
}
if (args is ["qwen-decode-ids", var tokenDirectory, var tokenIds])
{
    QwenTools.DecodeIds(tokenDirectory, tokenIds);
    return 0;
}
if (args is ["qwen-batch", var batchModel, var batchRoot, var batchSplit, var batchOutput, var batchBegin, var batchCount])
{
    QwenTools.Batch(batchModel, batchRoot, batchSplit, batchOutput, int.Parse(batchBegin, System.Globalization.CultureInfo.InvariantCulture), int.Parse(batchCount, System.Globalization.CultureInfo.InvariantCulture));
    return 0;
}
if (args is ["qwen-probe-decode", var decodeModel, var decodeProbe])
{
    QwenTools.ProbeDecoder(decodeModel, decodeProbe);
    return 0;
}
if (args is ["qwen-probe-encoder", var probeModel, var encoderProbe])
{
    QwenTools.ProbeEncoder(probeModel, encoderProbe);
    return 0;
}
if (args is ["qwen-convert-encoder", var qwenInput])
{
    QwenTools.ConvertEncoder(qwenInput);
    return 0;
}
if (args is ["qwen-positions", var positionProbe])
{
    QwenTools.Positions(positionProbe);
    return 0;
}
if (args is ["qwen-fetch", var qwenName, var qwenDirectory])
{
    await QwenTools.FetchAsync(qwenName, qwenDirectory, CancellationToken.None).ConfigureAwait(false);
    return 0;
}
if (args is ["onemodel-allocation-repro", var reproOutput])
{
    MoonshineAllocationProfile.Run(null, reproOutput);
    return 0;
}
if (args is ["moonshine-allocation-profile", var profileModel, var profileOutput])
{
    MoonshineAllocationProfile.Run(profileModel, profileOutput);
    return 0;
}
if (args is ["weight-hash", var hashPath])
{
    using var input = File.OpenRead(hashPath);
    Console.WriteLine($"{input.Length} {Convert.ToHexStringLower(SHA256.HashData(input))}");
    return 0;
}
if (args is ["mfa-fetch", var mfaDirectory])
{
    await MfaAssets.FetchAsync(mfaDirectory, CancellationToken.None);
    return 0;
}
if (args is ["omni-parity", var parityData, var parityNative, var parityReference, var parityOutput])
{
    OmniParity.Score(parityData, parityNative, parityReference, parityOutput);
    return 0;
}
if (args is ["omni-batch", var batchGraph, var batchTokenizer, var batchDirectory, var batchDestination, var omniBatchCount])
{
    OmniProbe.RunBatch(batchGraph, batchTokenizer, batchDirectory, batchDestination, int.Parse(omniBatchCount, System.Globalization.CultureInfo.InvariantCulture));
    return 0;
}
if (args is ["omni-probe", var probeGraph, var probeTokenizer, var probeAudio])
{
    OmniProbe.Run(probeGraph, probeTokenizer, probeAudio);
    return 0;
}
if (args is ["omni-convert", var omniInput, var omniOutput])
{
    using var input = File.OpenRead(omniInput);
    new OmniAsrConverter(TensorCheckpoint.Read(input, largeModel: true)).Save(omniOutput);
    using var converted = File.OpenRead(omniOutput);
    Console.WriteLine($"{converted.Length} {Convert.ToHexStringLower(SHA256.HashData(converted))}");
    return 0;
}
if (args is ["omni-tokenizer", var omniTokenizer])
{
    var data = OneSpeech.Text.SentencePieceProto.ReadFile(omniTokenizer);
    Console.WriteLine($"type={data.ModelType} pieces={data.Pieces.Length} normalizer={data.NormalizerName} charsmap={data.Charsmap.Length} dummy={data.AddDummyPrefix} strip={data.RemoveExtraWhitespaces} escape={data.EscapeWhitespaces} unk={data.UnkId}");
    for (int i = 0; i < Math.Min(40, data.Pieces.Length); i++) Console.WriteLine($"{i}: {data.Pieces[i]} type={data.Types[i]}");
    return 0;
}
if (args is ["omni-fetch", var omniRoot])
{
    await OmniTools.FetchAsync(omniRoot, CancellationToken.None);
    return 0;
}
if (args is ["omni-inspect", var omniCheckpoint])
{
    using var input = File.OpenRead(omniCheckpoint);
    foreach (var pair in TensorCheckpoint.Read(input, largeModel: true))
        Console.WriteLine($"{pair.Key} [{string.Join(",", pair.Value.Shape)}]");
    return 0;
}
if (args is ["moonshine-service-smoke", var publisherDirectory, var freshCache, var smokeAudio])
{
    await MoonshineServiceSmoke.Run(publisherDirectory, freshCache, smokeAudio);
    return 0;
}
if (args is ["moonshine-refresh-incomplete", var refreshModel, var refreshSplit, var refreshOutput])
{
    MoonshineBatch.Run(refreshModel, refreshSplit, refreshOutput, 0, 200, refreshIncomplete: true);
    return 0;
}
if (args is ["moonshine-validate-saved", var savedModel, var savedOurs, var savedReference])
{
    MoonshineBatch.ValidateSaved(savedModel, savedOurs, savedReference);
    return 0;
}
if (args is ["moonshine-edges", var meOutput])
{
    MoonshineProbe.Edges(meOutput);
    return 0;
}
if (args is ["moonshine-bakeoff-metrics", var bakeModel, var bakeClip, var bakeOutput, var bakeThreads])
{
    MoonshineLiveMetrics.Run(bakeModel, bakeClip, bakeOutput, int.Parse(bakeThreads, System.Globalization.CultureInfo.InvariantCulture), sixtySeconds: true);
    return 0;
}
if (args is ["moonshine-indicative", var indicativeDirectory, var indicativeAudio, var indicativeOutput])
{
    MoonshineIndicative.Run(indicativeDirectory, indicativeAudio, indicativeOutput);
    return 0;
}
if (args is ["moonshine-export-slice", var esRoot, var esOurs, var esReference, var esPrefix, var esCount])
{
    MoonshineBatch.Export(esRoot, esOurs, esReference, esPrefix, int.Parse(esCount, System.Globalization.CultureInfo.InvariantCulture));
    return 0;
}
if (args is ["moonshine-latency", var latencyHypotheses, var latencyTiming, var latencyOutput])
{
    MoonshineLatency.Run(latencyHypotheses, latencyTiming, latencyOutput);
    return 0;
}
if (args is ["moonshine-tokenizer-corpus", var tcDirectory, var tcPath])
{
    MoonshineTools.TokenizerCorpus(tcDirectory, tcPath);
    return 0;
}
if (args is ["moonshine-live-probe", var mlDirectory, var mlAudio])
{
    MoonshineLiveProbe.Check(mlDirectory, mlAudio);
    return 0;
}
if (args is ["moonshine-live-batch", var mlModel, var mlSplit, var mlOutput, var mlBegin, var mlCount, var mlChunk])
{
    MoonshineLiveProbe.Batch(mlModel, mlSplit, mlOutput, int.Parse(mlBegin, System.Globalization.CultureInfo.InvariantCulture), int.Parse(mlCount, System.Globalization.CultureInfo.InvariantCulture), int.Parse(mlChunk, System.Globalization.CultureInfo.InvariantCulture));
    return 0;
}
if (args is ["moonshine-reproduce", var mrCatalog, var mrRoot])
{
    MoonshineTools.Reproduce(mrCatalog, mrRoot);
    return 0;
}
if (args is ["moonshine-catalog", var mcRoot, var mcOutput])
{
    MoonshineTools.Catalog(mcRoot, mcOutput);
    return 0;
}
if (args is ["moonshine-batch", var mbModel, var mbSplit, var mbOutput, var mbBegin, var mbCount])
{
    MoonshineBatch.Run(mbModel, mbSplit, mbOutput, int.Parse(mbBegin, System.Globalization.CultureInfo.InvariantCulture), int.Parse(mbCount, System.Globalization.CultureInfo.InvariantCulture));
    return 0;
}
if (args is ["moonshine-export", var mxRoot, var mxOurs, var mxReference, var mxPrefix])
{
    MoonshineBatch.Export(mxRoot, mxOurs, mxReference, mxPrefix);
    return 0;
}
if (args is ["moonshine-frontend-probe", var frontendDirectory, var frontendReference])
{
    MoonshineFrontendProbe.Run(frontendDirectory, frontendReference);
    return 0;
}
if (args is ["moonshine-tokenizer", var mtDirectory])
{
    var mt = new OneSpeech.Moonshine.MoonshineTokenizer(mtDirectory);
    foreach (string text in new[] { "", " hello world", "a  b\nC", "café e\u0301", "中文 🙂", "<s></s>" })
    {
        var ids = mt.Encode(text);
        Console.WriteLine(string.Join(",", ids) + " | " + mt.Decode(ids, false));
    }
    return 0;
}
if (args is ["moonshine-transcribe", var moonshineModel, var moonshineAudio])
{
    using var mr = OneSpeech.Moonshine.MoonshineRecognizer.Load(moonshineModel);
    mr.TokenTrace = ids => Console.WriteLine(string.Join(",", ids));
    var ma = OneAudio.Audio.Load(moonshineAudio, new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 });
    Console.WriteLine(mr.Transcribe(new OneSpeech.AudioBuffer(ma.Samples, 16000), "en", CancellationToken.None).Text);
    return 0;
}
if (args is ["moonshine-attention-probe", var attentionDirectory, var attentionReference])
{
    MoonshineAttentionProbe.Run(attentionDirectory, attentionReference);
    return 0;
}
if (args is ["moonshine-probe", var moonshineGraphDirectory, var moonshineProbeDirectory])
{
    MoonshineProbe.Run(moonshineGraphDirectory, moonshineProbeDirectory);
    return 0;
}
if (args.Length is 2 or 3 && args[0] is "moonshine-convert" or "moonshine-convert-debug")
{
    string moonshineSource = args[1];
    string moonshineDestination = args.Length == 3 ? args[2] : moonshineSource;
    Directory.CreateDirectory(moonshineDestination);
    if (moonshineSource != moonshineDestination) File.Copy(Path.Combine(moonshineSource, "config.json"), Path.Combine(moonshineDestination, "config.json"), overwrite: true);
    _ = OneSpeech.Moonshine.MoonshineProvenance.Verify(moonshineSource, converted: false, CancellationToken.None);
    var mc = OneSpeech.Moonshine.MoonshineConfiguration.Load(moonshineSource);
    using var mi = File.OpenRead(Path.Combine(moonshineSource, "model.safetensors"));
    using var mw = new SafeTensorFile(mi);
    var me = new MoonshineGraphWriter(mw, diagnostics: args[0].EndsWith("debug", StringComparison.Ordinal));
    me.Encoder(mc); me.Save(Path.Combine(moonshineDestination, "encoder.onemodel"), CancellationToken.None);
    var md = new MoonshineGraphWriter(mw, diagnostics: args[0].EndsWith("debug", StringComparison.Ordinal));
    md.Decoder(mc); md.Save(Path.Combine(moonshineDestination, "decoder.onemodel"), CancellationToken.None);
    if (mc.Streaming)
    {
        var project = new MoonshineGraphWriter(mw); project.DecoderProject(mc); project.Save(Path.Combine(moonshineDestination, "decoder-project.onemodel"), CancellationToken.None);
        var step = new MoonshineGraphWriter(mw); step.DecoderStep(mc); step.Save(Path.Combine(moonshineDestination, "decoder-step.onemodel"), CancellationToken.None);
    }
    return 0;
}
if (args is ["moonshine-fetch", var moonshineName, var moonshineDirectory])
{
    await MoonshineTools.FetchAsync(moonshineName, moonshineDirectory, CancellationToken.None).ConfigureAwait(false);
    return 0;
}
if (args is ["moonshine-inspect", var moonshineInput])
{
    MoonshineTools.Inspect(moonshineInput);
    return 0;
}
if (args is ["speaker-extract-ecapa", var ecapaGraph, var ecapaData, var ecapaIds, var ecapaDestination, var ecapaStart, var ecapaCount])
{
    SpeakerBatch.Extract(ecapaGraph, ecapaData, ecapaIds, ecapaDestination, int.Parse(ecapaStart, System.Globalization.CultureInfo.InvariantCulture), int.Parse(ecapaCount, System.Globalization.CultureInfo.InvariantCulture), 192, ecapa: true);
    return 0;
}
if (args is ["speaker-probe-ecapa", var ecapaProbeGraph, var ecapaAudio, var ecapaProbeOutput])
{
    SpeakerProbe.RunEcapa(ecapaProbeGraph, ecapaAudio, ecapaProbeOutput);
    return 0;
}
if (args is ["speaker-convert-ecapa", var ecapaInput, var ecapaOutput])
{
    using var input = File.OpenRead(ecapaInput);
    new EcapaConverter(TensorCheckpoint.Read(input)).Save(ecapaOutput);
    return 0;
}
if (args is ["speaker-ids", var idsTrials, var idsClean, var idsOutput, var idsParity])
{
    SpeakerBatch.PrepareIds(idsTrials, idsClean, idsOutput, idsParity);
    return 0;
}
if (args is ["speaker-extract-camp", var extractGraph, var extractData, var extractIds, var extractOutput, var extractStart, var extractCount, var extractDimension])
{
    SpeakerBatch.Extract(extractGraph, extractData, extractIds, extractOutput, int.Parse(extractStart, System.Globalization.CultureInfo.InvariantCulture), int.Parse(extractCount, System.Globalization.CultureInfo.InvariantCulture), int.Parse(extractDimension, System.Globalization.CultureInfo.InvariantCulture));
    return 0;
}
if (args is ["speaker-run-features", var featureGraph, var featureInput, var vectorOutput])
{
    SpeakerProbe.RunFeatures(featureGraph, featureInput, vectorOutput);
    return 0;
}
if (args is ["speaker-trials", var cleanRoot, var otherRoot, var trialsOutput, var seed])
{
    SpeakerVerification.Generate(cleanRoot, otherRoot, trialsOutput, int.Parse(seed, System.Globalization.CultureInfo.InvariantCulture));
    return 0;
}
if (args is ["speaker-score", var trialFile, var oursDirectory, var referenceDirectory, var scoreOutput])
{
    SpeakerVerification.ScoreFiles(trialFile, oursDirectory, referenceDirectory, scoreOutput);
    return 0;
}
if (args is ["speaker-probe", var graphPath, var audioPath, var probeOutput])
{
    SpeakerProbe.Run(graphPath, audioPath, probeOutput);
    return 0;
}
if (args is ["speaker-convert-camp", var campCheckpoint, var campOutput])
{
    using var campInput = File.OpenRead(campCheckpoint);
    new CampPlusConverter(TensorCheckpoint.Read(campInput)).Save(campOutput);
    return 0;
}
if (args is ["speaker-fetch", var root])
{
    await SpeakerTools.FetchAsync(root, CancellationToken.None).ConfigureAwait(false);
    return 0;
}
if (args is ["pyannote-diarize", var pyannoteSnapshot, var pyannoteModelCache, var pyannoteEmbedder, var pyannoteAudio, var pyannoteRttm])
{
    await PyannoteDiarizationProbe.Run(pyannoteSnapshot, pyannoteModelCache, pyannoteEmbedder, pyannoteAudio, pyannoteRttm);
    return 0;
}
if (args is ["pyannote-probe", var pyannoteCheckpoint, var pyannoteCache, var pyannoteProbes])
{
    PyannoteProbe.Run(pyannoteCheckpoint, pyannoteCache, pyannoteProbes);
    return 0;
}
if (args is ["pyannote-inspect", var segmentationCheckpoint])
{
    try
    {
        using var input = File.OpenRead(segmentationCheckpoint);
        foreach (var pair in TensorCheckpoint.Read(input, segmentation: true))
            Console.WriteLine(pair.Key + " " + string.Join(",", pair.Value.Shape));
        return 0;
    }
    catch (InvalidDataException error) { Console.Error.WriteLine(error.Message); return 1; }
    catch (IOException error) { Console.Error.WriteLine(error.Message); return 1; }
}
if (args is ["speaker-inspect", var checkpoint])
{
    using var input = File.OpenRead(checkpoint);
    foreach (var pair in TensorCheckpoint.Read(input))
        Console.WriteLine($"{pair.Key} [{string.Join(',', pair.Value.Shape)}]");
    return 0;
}

if (args.Length < 2 || args[0] is not ("fetch" or "whisper" or "whisper-debug" or "norm-probe" or "block-probe"))
{
    Console.Error.WriteLine("Usage: OneSpeech.Convert fetch <tiny|base|small|large-v3|large-v3-turbo> <directory> | whisper <directory> | speaker-<command> ...");
    return 2;
}
using var cancellation = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; cancellation.Cancel(); };
if (args[0] == "fetch")
{
    if (args.Length != 3) return 2;
    var pin = args[1] switch
    {
        "tiny" => ("169d4a4341b33bc18d8881c4b69c2e104e1cc0af", "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395"),
        "base" => ("e37978b90ca9030d5170a5c07aadb050351a65bb", "07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b"),
        "large-v3" => ("06f233fe06e710322aca913c1bc4249a0d71fce1", "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95"),
        "small" => ("973afd24965f72e36ca33b3055d56a652f456b4d", "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b"),
        "large-v3-turbo" => ("41f01f3fe87f28c78e2fbf8b568835947dd65ed9", "542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1"),
        _ => throw new ArgumentException("Only explicitly pinned checkpoints are accepted.")
    };
    Directory.CreateDirectory(args[2]);
    using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
    foreach (string name in new[] { "config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json", "vocab.json", "merges.txt", "added_tokens.json", "tokenizer_config.json", "special_tokens_map.json", "normalizer.json", "model.safetensors" })
    {
        string path = Path.Combine(args[2], name);
        if (!File.Exists(path))
        {
            using var response = await http.GetAsync($"https://huggingface.co/openai/whisper-{args[1]}/resolve/{pin.Item1}/{name}", HttpCompletionOption.ResponseHeadersRead, cancellation.Token);
            response.EnsureSuccessStatusCode();
            await using (var output = File.Create(path + ".part")) await response.Content.CopyToAsync(output, cancellation.Token);
            File.Move(path + ".part", path);
        }
        await using var input = File.OpenRead(path);
        string sha = Convert.ToHexStringLower(await SHA256.HashDataAsync(input, cancellation.Token));
        if (name == "model.safetensors" && sha != pin.Item2) throw new InvalidDataException("Upstream weight digest mismatch.");
        Console.WriteLine($"{name} {input.Length} {sha}");
    }
    File.WriteAllText(Path.Combine(args[2], "upstream.txt"), $"openai/whisper-{args[1]}\n{pin.Item1}\n{pin.Item2}\n");
    return 0;
}
var config = WhisperConfiguration.Load(args[1]);
string weightPath = Path.Combine(args[1], "model.safetensors");
string pinPath = Path.Combine(args[1], "upstream.txt");
if (!File.Exists(pinPath)) throw new InvalidDataException("Run fetch first to establish verified upstream provenance.");
string[] provenance = WhisperCheckpoint.Read(args[1]);
_ = WhisperCheckpoint.Provenance(provenance);
using (var input = File.OpenRead(weightPath))
    if (Convert.ToHexStringLower(SHA256.HashData(input)) != provenance[2]) throw new InvalidDataException("Weight digest mismatch before conversion.");
using var weights = new SafeTensorFile(File.OpenRead(weightPath));
if (args[0] == "block-probe")
{
    int layer = config.EncoderLayers == 12 ? 6 : 20;
    var block = new WhisperGraphWriter(weights);
    block.Encoder(config, layer, layer + 1);
    block.Save(Path.Combine(args[1], "block-probe.onemodel"), cancellation.Token);
    return 0;
}
if (args[0] == "norm-probe")
{
    var norm = new WhisperGraphWriter(weights);
    norm.NormalizationProbe(config);
    norm.Save(Path.Combine(args[1], "norm-probe.onemodel"), cancellation.Token);
    return 0;
}
foreach (string graph in config.EncoderLayers > 24 ? new[] { "encoder-part0", "encoder", "decoder" } : new[] { "encoder", "decoder" })
{
    var builder = new WhisperGraphWriter(weights, args[0] == "whisper-debug");
    if (graph == "encoder-part0") builder.Encoder(config, 0, 16);
    else if (graph == "encoder") builder.Encoder(config, config.EncoderLayers > 24 ? 16 : 0);
    else builder.Decoder(config);
    string path = Path.Combine(args[1], graph + (args[0] == "whisper-debug" ? ".debug.onemodel" : ".onemodel"));
    builder.Save(path, cancellation.Token);
    using var file = File.OpenRead(path);
    Console.WriteLine($"{graph}.onemodel {file.Length} {Convert.ToHexStringLower(SHA256.HashData(file))}");
}
return 0;
