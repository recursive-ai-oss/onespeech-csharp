using OneSpeech;
using OneSpeech.Audio;
using OneSpeech.Catalog;
using OneSpeech.Moonshine;

if (args.Length != 2) { Console.Error.WriteLine("Usage: offline-package-smoke <audio> <empty-cache-directory>"); return 2; }
var store = new ModelStore(args[1], new NoNetwork());
string? tiny = null;
foreach (string id in new[] { "iic/speech_campplus_sv_en_voxceleb_16k", "moonshine-ai/moonshine-tiny", "moonshine-ai/moonshine-base", "moonshine-ai/moonshine-streaming-tiny" })
{
    var model = await store.PullAsync(id);
    Console.WriteLine("Offline verified and converted: " + id);
    if (id == "moonshine-ai/moonshine-tiny") tiny = model.Directory;
}
using var input = AudioFileSource.Open(args[0], CancellationToken.None);
var samples = new List<float>();
var block = new float[16000 * input.Channels];
int count;
while ((count = input.Read(block, CancellationToken.None)) != 0)
{
    if (samples.Count + count > 60 * 16000) throw new InvalidOperationException("Smoke input must be at most 60 seconds.");
    for (int i = 0; i < count * input.Channels; i += input.Channels)
    {
        float value = 0;
        for (int c = 0; c < input.Channels; c++) value += block[i + c] / input.Channels;
        samples.Add(value);
    }
}
using var recognizer = MoonshineRecognizer.Load(tiny!);
var transcript = recognizer.Transcribe(new AudioBuffer(samples.ToArray(), 16000), "en", CancellationToken.None);
if (string.IsNullOrWhiteSpace(transcript.Text)) throw new InvalidOperationException("Expected speech in the smoke clip.");
Console.WriteLine(transcript.Text);
return 0;

sealed class NoNetwork : ModelSource
{
    public override ValueTask<Stream> OpenReadAsync(CatalogEntry model, CatalogFile file, CancellationToken cancellationToken) => throw new InvalidOperationException("Network is disabled: missing package file " + file.Name);
}
