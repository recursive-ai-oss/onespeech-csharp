using System.Collections.Immutable;
using System.Text;
using System.Text.Json;
using OneSpeech.Serialization;
using Xunit;

namespace OneSpeech.Tests;

public sealed class LongTranscriptTests
{
    [Fact]
    public void TwentyFourHoursOfDenseWordsSegmentsAndTurnsRoundTrip()
    {
        const int count = 24 * 60 * 60 * 6;
        var speaker = new SpeakerLabel { Id = "s1", Name = "Synthetic document" };
        var words = ImmutableArray.CreateBuilder<Word>(count);
        for (int i = 0; i < count; i++)
            words.Add(new Word { Text = i == 123 ? "caf\u00e9" : "spoken", Start = At(i), End = At(i + 1),
                Confidence = .9, TimingMethod = TimingMethod.Native, Speaker = speaker, Language = "en", Overlap = i % 17 == 0 });
        var all = words.MoveToImmutable();
        var segments = ImmutableArray.CreateBuilder<Segment>(count / 60);
        for (int i = 0; i < count; i += 60)
        {
            var group = all.Slice(i, 60);
            segments.Add(new Segment { Text = string.Join(' ', group.Select(w => w.Text)), Start = At(i), End = At(i + 60), Words = group, Language = "en" });
        }
        var turns = ImmutableArray.CreateBuilder<Turn>(count / 600);
        for (int i = 0; i < count; i += 600)
        {
            var group = all.Slice(i, 600);
            turns.Add(new Turn { Text = string.Join(' ', group.Select(w => w.Text)), Start = At(i), End = At(i + 600), Words = group, Speaker = speaker });
        }
        var document = new Transcript { Text = string.Join(' ', all.Select(w => w.Text)), Words = all,
            Segments = segments.MoveToImmutable(), Turns = turns.MoveToImmutable(), Duration = TimeSpan.FromHours(24), Language = "en" };
        string json = document.ToJson();
        Assert.True(Encoding.UTF8.GetByteCount(json) > 8 * 1024 * 1024);
        var restored = Transcript.FromJson(json);
        Assert.Equal(document, restored);
        Assert.Equal(count, restored.Words.Length);
        Assert.Equal(TimeSpan.FromHours(24), restored.Words[^1].End);
        TestContext.Current.TestOutputHelper!.WriteLine($"24 h: {count} words, {document.Segments.Length} segments, {document.Turns.Length} turns; {Encoding.UTF8.GetByteCount(json)} UTF-8 bytes.");

        static TimeSpan At(int word) => TimeSpan.FromTicks((long)word * TimeSpan.TicksPerSecond / 6);
    }

    [Fact]
    public void WriterRejectsExpandedByteBudgetWithoutRetainingTheOutput()
    {
        var repeated = new Word { Text = new string('x', 65536) };
        var document = new Transcript { Text = "", Words = [.. Enumerable.Repeat(repeated, 20_000)] };
        long before = GC.GetAllocatedBytesForCurrentThread();
        var error = Assert.Throws<InvalidTranscriptJsonException>(() => document.ToJson());
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Contains("1 GB", error.Message, StringComparison.Ordinal);
        Assert.InRange(allocated, 0, 32 * 1024 * 1024);
    }

    [Fact]
    public void TranscriptBoundsArePreflightedIndependentlyOfCatalogBounds()
    {
        string text = new('x', 65537);
        var document = new Transcript { Text = text,
            Segments = [new Segment { Text = text }],
            Turns = [new Turn { Text = text, Speaker = SpeakerLabel.Unassigned, Start = TimeSpan.Zero, End = TimeSpan.FromHours(24) }] };
        Assert.Equal(document, Transcript.FromJson(document.ToJson()));
        Assert.Throws<InvalidTranscriptJsonException>(() => (document with { Words = [new Word { Text = text }] }).ToJson());
        Assert.Throws<JsonException>(() => JsonBounds.Check(Encoding.UTF8.GetBytes("{\"text\":\"" + text + "\"}")));
        Assert.Throws<InvalidTranscriptJsonException>(() => (document with { Duration = TimeSpan.FromHours(24) + TimeSpan.FromTicks(1) }).ToJson());
        Assert.Throws<InvalidTranscriptJsonException>(() => Transcript.FromJson("{\"schemaVersion\":1,\"transcript\":{\"text\":\"\",\"duration\":86400.1}}"));
        byte[] oversizedArray = Encoding.UTF8.GetBytes("{\"schemaVersion\":1,\"transcript\":{\"text\":\"\",\"words\":[" + string.Join(',', Enumerable.Repeat("null", TranscriptJsonBounds.MaxItems + 1)) + "]}}");
        long allocated = GC.GetAllocatedBytesForCurrentThread();
        var error = Assert.Throws<InvalidTranscriptJsonException>(() => Transcript.FromJson(oversizedArray));
        long rejectedBytes = GC.GetAllocatedBytesForCurrentThread() - allocated;
        Assert.Contains("array", error.Message, StringComparison.Ordinal);
        Assert.InRange(rejectedBytes, 0, 256 * 1024);
        Assert.Throws<InvalidTranscriptJsonException>(() => (document with { Words = default }).ToJson());
        Assert.Throws<InvalidTranscriptJsonException>(() => (document with { Words = [new Word { Text = "x", Start = TimeSpan.Zero, End = TimeSpan.FromHours(25) }] }).ToJson());
        Assert.Throws<InvalidTranscriptJsonException>(() => Transcript.FromJson("{\"schemaVersion\":1,\"transcript\":{\"text\":\"\",\"words\":[{\"text\":\"" + text + "\"}]}}"));
        var reader = new Utf8JsonReader(oversizedArray); reader.Read();
        try
        {
            _ = new TranscriptJsonConverter().Read(ref reader, typeof(Transcript), JsonSerializerOptions.Default);
            Assert.Fail("The converter must preflight array limits too.");
        }
        catch (JsonException exception) { Assert.Contains("array", exception.Message, StringComparison.Ordinal); }
        var unadvanced = new Utf8JsonReader(oversizedArray);
        try
        {
            _ = new TranscriptJsonConverter().Read(ref unadvanced, typeof(Transcript), JsonSerializerOptions.Default);
            Assert.Fail("An unadvanced converter reader must preflight too.");
        }
        catch (JsonException exception) { Assert.Contains("array", exception.Message, StringComparison.Ordinal); }
    }
}
