using System.Collections.Immutable;

namespace OneSpeech;

internal static class SequenceValue
{
    internal static bool Equals<T>(ImmutableArray<T> a, ImmutableArray<T> b) =>
        a.IsDefault || b.IsDefault ? a.IsDefault == b.IsDefault : a.SequenceEqual(b);
    internal static int Hash<T>(ImmutableArray<T> values)
    {
        var hash = new HashCode();
        if (!values.IsDefault) foreach (var value in values) hash.Add(value);
        return hash.ToHashCode();
    }
}
