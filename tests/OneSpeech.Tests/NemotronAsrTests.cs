using System.Text;
using System.Text.Json;
using OneSpeech.Catalog;
using OneSpeech.FastConformer;
using OneSpeech.Live;
using Xunit;

namespace OneSpeech.Tests;

public sealed class NemotronAsrTests
{
    internal static ModelInfo Info()
    {
        var e = ModelCatalog.Default.Get(NemotronAsrRecognizer.Identifier);
        return new() { Id = e.Id, Revision = e.Revision, Source = e.Source, Tier = e.Tier, License = e.License, Attribution = e.Attribution, Files = [.. e.Files.Concat(e.ConvertedFiles).Select(f => new ModelFileInfo { Name = f.Name, Sha256 = f.Sha256 })] };
    }
    [Fact]
    public void CatalogIsPublisherOnlyAndCarriesFullAgreement()
    {
        var e = ModelCatalog.Default.Get(NemotronAsrRecognizer.Identifier);
        Assert.Equal(ModelTier.Tier2, e.Tier); Assert.False(e.IsDefault); Assert.False(e.Gated); Assert.True(e.Capabilities.Streaming);
        Assert.Equal("OpenMDW-1.1", e.License); Assert.Contains("Developed by NVIDIA", e.Attribution, StringComparison.Ordinal);
        Assert.Contains("OpenMDW License Agreement, version 1.1", e.Attribution, StringComparison.Ordinal);
        Assert.Contains("If you file, maintain, or voluntarily participate", e.Attribution, StringComparison.Ordinal);
        Assert.Contains("IN NO EVENT SHALL", e.Attribution, StringComparison.Ordinal);
        Assert.All(e.Files.Concat(e.ConvertedFiles), f => { Assert.Null(f.MirrorUrl); Assert.StartsWith(e.Source + "/resolve/" + e.Revision + "/", f.UpstreamUrl, StringComparison.Ordinal); });
        Assert.Throws<InvalidModelCatalogException>(() => new ModelCatalog([e with { IsDefault = true }]));
        Assert.Throws<InvalidModelCatalogException>(() => new ModelCatalog([e with { ConvertedFiles = [e.ConvertedFiles[0] with { MirrorUrl = "https://example.org/forbidden" }] }]));
    }
    [Theory]
    [InlineData(160, 0)]
    [InlineData(0, 0)]
    [InlineData(80, -1)]
    public void InvalidOptionsAreRefusedBeforeOpeningModelFiles(int chunk, int threads)
        => Assert.Throws<ArgumentOutOfRangeException>(() => NemotronAsrRecognizer.Load(new("missing", Info()), new() { ChunkMilliseconds = chunk, Threads = threads }, TestContext.Current.CancellationToken));
    [Fact]
    public void VocabularySkipsOnlyDeclaredSpecialsAndPreservesRnntRepeats()
    {
        using var memory = new MemoryStream();
        using (var writer = new Utf8JsonWriter(memory))
        {
            writer.WriteStartObject(); writer.WriteStartArray("added_tokens");
            foreach (int id in new[] { 0, 13087, 13088 }) { writer.WriteStartObject(); writer.WriteNumber("id", id); writer.WriteBoolean("special", true); writer.WriteEndObject(); }
            writer.WriteEndArray(); writer.WriteStartObject("model"); writer.WriteString("type", "BPE"); writer.WriteStartObject("vocab");
            for (int i = 0; i < 13087; i++) writer.WriteNumber(i == 0 ? "<unk>" : i == 1 ? "▁Hello" : i == 2 ? "▁" : "v" + i, i);
            writer.WriteEndObject(); writer.WriteEndObject(); writer.WriteEndObject();
        }
        var vocabulary = new ParakeetVocabulary(memory.ToArray(), 13087);
        Assert.Equal("Hello Hello", vocabulary.Decode([1, 1])); Assert.Equal("  ", vocabulary.Decode([2, 2, 2]));
        Assert.Equal("Hello", vocabulary.Decode([0, 1, 13087, 13088]));
        Assert.Throws<InvalidDataException>(() => vocabulary.Decode([13089]));
        Assert.Throws<InvalidDataException>(() => new ParakeetVocabulary(Encoding.UTF8.GetBytes("{}"), 13087));
    }
    [Theory]
    [InlineData(80)]
    [InlineData(560)]
    public void RealModelLivePartialsAreRevisionedAndFinalMatchesWholeInput(int chunk)
    {
        string? directory = Environment.GetEnvironmentVariable("ONESPEECH_NEMOTRON_DIRECTORY"); if (directory is null) return;
        var ct = TestContext.Current.CancellationToken;
        using var model = NemotronAsrRecognizer.Load(new(directory, Info()), new() { ChunkMilliseconds = chunk, Threads = 4 }, ct);
        var clip = OneAudio.Audio.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "1089-134686-0001.flac"), new OneAudio.LoadOptions { SampleRate = 16000, Channels = 1 }, ct);
        var audio = new AudioBuffer(clip.Samples, 16000);
        Assert.NotEmpty(model.Transcribe(audio, "en", ct).Words);
        using var whole = new NemotronAsrState(model, 0, ct);
        var expected = whole.Decode(clip.Samples, "en", true, TimeSpan.Zero, audio.Duration, ct);
        using var session = new UtteranceLiveSession(model, (_, _) => 1, new NemotronAsrState(model, 0, ct), "en");
        List<LiveEvent> updates = []; List<(PartialEvent Event, string Text)> snapshots = [];
        for (int offset = 0; offset < clip.Samples.Length; offset += 1600)
        {
            foreach (var update in session.Push(new AudioBuffer(clip.Samples.Slice(offset, Math.Min(1600, clip.Samples.Length - offset)), 16000), ct))
            {
                updates.Add(update); if (update is PartialEvent partial) snapshots.Add((partial, partial.Segment.Text));
            }
        }
        updates.AddRange(session.Complete(ct)); Assert.NotEmpty(snapshots);
        var final = Assert.Single(updates.OfType<FinalEvent>()); Assert.Equal(expected.Text, final.Segment.Text);
        Assert.All(snapshots, snapshot => Assert.Equal(snapshot.Text, snapshot.Event.Segment.Text));
        var revisions = updates.OfType<PartialEvent>().Select(x => x.Revision).ToArray(); Assert.Equal(revisions.Order(), revisions); Assert.Equal(revisions.Length, revisions.Distinct().Count());
        Assert.All(updates, update => Assert.True(update.AudioTime <= audio.Duration));
        Assert.All(final.Segment.Words, word => { Assert.NotNull(word.Start); Assert.NotNull(word.End); Assert.InRange(word.Confidence!.Value, 0, 1); Assert.True(word.End <= audio.Duration); });
        Assert.Throws<CapabilityNotSupportedException>(() => model.Transcribe(audio, "xx", ct));
        if (chunk == 560) Assert.Equal("en", model.DetectLanguage(audio, ct));
    }
}
