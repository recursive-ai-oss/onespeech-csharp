using Xunit;

namespace OneSpeech.Tests;

public sealed class SpeakerVerificationTests
{
    [Fact]
    public void RocHandlesTiesAndKnownOperatingPoints()
    {
        var perfect = SpeakerVerification.Measure([(true, .9), (true, .8), (false, .2), (false, .1)]);
        Assert.Equal(0, perfect.Eer); Assert.Equal(0, perfect.MinDcf);
        var tied = SpeakerVerification.Measure([(true, .5), (false, .5)]);
        Assert.Equal(.5, tied.Eer); Assert.Equal(1, tied.MinDcf);
        var mixed = SpeakerVerification.Measure([(true, .9), (false, .8), (true, .7), (false, .6)]);
        Assert.Equal(.5, mixed.Eer); Assert.Equal(.5, mixed.MinDcf);
        Assert.Throws<ArgumentException>(() => SpeakerVerification.Measure([(true, .1)]));
        Assert.Throws<ArgumentException>(() => SpeakerVerification.Measure([(true, double.NaN), (false, .1)]));
    }
    [Fact]
    public void TrialsAreBalancedUniqueAndReproducible()
    {
        string[][] speakers = [["1-1-1", "1-1-2", "1-1-3"], ["2-1-1", "2-1-2", "2-1-3"], ["3-1-1", "3-1-2", "3-1-3"]];
        var a = SpeakerVerification.MakeTrials(speakers, 7, 1729);
        var b = SpeakerVerification.MakeTrials(speakers, 7, 1729);
        Assert.Equal(a, b);
        Assert.Equal(14, a.Length);
        Assert.Equal(7, a.Count(x => x.Same));
        Assert.Equal(14, a.Select(x => (x.First, x.Second)).Distinct().Count());
        foreach (var t in a) Assert.Equal(t.Same, t.First.Split('-')[0] == t.Second.Split('-')[0]);
    }
}
