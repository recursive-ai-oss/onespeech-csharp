using Xunit;

namespace OneSpeech.Tests;

[Collection("Graph timing")]
public sealed class GraphTimingTests
{
    [Fact]
    public void ConditionalCallsAndThreadOverrideFollowBuildFlag()
    {
        GraphTiming.Reset();
        int threads = 1;
        GraphTiming.ThreadCount = 3;
        GraphTiming.OverrideThreads(ref threads);
        Assert.Equal(GraphTiming.IsCompiled ? 3 : 1, threads);
        int arguments = 0;
        int Category() { arguments++; return 7; }
        GraphTiming.Begin(Category());
        GraphTiming.BeginCost(4);
        var allocation = new byte[8192];
        GraphTiming.EndCost(4);
        GraphTiming.End();
        Assert.Equal(GraphTiming.IsCompiled ? 1 : 0, arguments);
        Assert.Equal(GraphTiming.IsCompiled, GraphTiming.CostSnapshot()[4] > 0);
        GC.KeepAlive(allocation);
        GraphTiming.BeginOptional(null, 6);
        GraphTiming.EndOptional(null);
        Assert.Equal(0, GraphTiming.Snapshot()[6].Calls);
        var graph = new object();
        GraphTiming.BeginOptional(graph, 6);
        GraphTiming.EndOptional(graph);
        Assert.Equal(GraphTiming.IsCompiled ? 1 : 0, GraphTiming.Snapshot()[6].Calls);
        var sample = GraphTiming.Snapshot()[7];
        Assert.Equal(GraphTiming.IsCompiled ? 1 : 0, sample.Calls);
        if (GraphTiming.IsCompiled)
        {
            Assert.True(sample.Ticks > 0);
            Assert.True(sample.Bytes >= 8192);
        }
        var features = new OneSpeech.Audio.WhisperLogMelSpectrogram(128);
        var audio = new AudioBuffer(new float[16000], 16000);
        GraphTiming.Reset();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        _ = features.ExtractContext(audio, 0, float.NegativeInfinity, TestContext.Current.CancellationToken);
        _ = features.ExtractQwen(audio, TestContext.Current.CancellationToken);
        clock.Stop();
        long featureTicks = GraphTiming.CostSnapshot()[1];
        Assert.True(featureTicks <= clock.ElapsedTicks);
        Assert.Equal(GraphTiming.IsCompiled, featureTicks > 0);
        GraphTiming.ThreadCount = 0;
        GraphTiming.Reset();
    }
}

[CollectionDefinition("Graph timing", DisableParallelization = true)]
public sealed class GraphTimingIsolation;
