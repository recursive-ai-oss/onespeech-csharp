using OneSpeech.Catalog;
using Xunit;

namespace OneSpeech.Tests;

public sealed class RoleDefaultTests
{
    [Fact]
    public void RolePickIsIndependentOfFamilyAndRoundTrips()
    {
        var catalog = ModelCatalog.Default;
        Assert.Equal("openai/whisper-large-v3-turbo", catalog.GetRoleDefault("transcription").Id);
        var qwen = catalog.Get("Qwen/Qwen3-ASR-1.7B") with { DefaultRoles = ["transcription"] };
        var moved = new ModelCatalog(catalog.Entries.Select(e => e.Id == qwen.Id ? qwen : e with { DefaultRoles = [] }));
        Assert.Equal(qwen.Id, moved.GetRoleDefault("transcription").Id);
        Assert.Equal("openai/whisper-large-v3-turbo", moved.GetDefault("whisper").Id);
        Assert.Throws<ModelNotFoundException>(() => moved.GetRoleDefault("unknown"));
        Assert.Equal(qwen.DefaultRoles, CatalogEntry.FromJson(System.Text.Json.JsonSerializer.SerializeToUtf8Bytes(qwen, OneSpeech.Serialization.SpeechJsonContext.Default.CatalogEntry)).DefaultRoles);
    }

    [Fact]
    public void DuplicateAndTierTwoRoleDefaultsAreRejected()
    {
        var entries = ModelCatalog.Default.Entries;
        Assert.Throws<InvalidModelCatalogException>(() => new ModelCatalog(entries.Select(e => e.Id == "Qwen/Qwen3-ASR-1.7B" ? e with { DefaultRoles = ["transcription"] } : e)));
        var tierTwo = entries.First(e => e.Tier == ModelTier.Tier2);
        Assert.Throws<InvalidModelCatalogException>(() => new ModelCatalog([tierTwo with { DefaultRoles = ["transcription"] }]));
        Assert.Throws<InvalidModelCatalogException>(() => new ModelCatalog([entries[0] with { DefaultRoles = ["transcription", "transcription"] }]));
    }
}
