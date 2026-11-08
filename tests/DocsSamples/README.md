# Documentation sample checks

Run `dotnet run --project tests/DocsSamples -- .` from the repository root after restore/build. This checks Markdown links to local files and headings, verifies that every C# fence in the README and guides is present in `Examples.cs`, and executes the model-free JSON example. External URLs are not fetched by the check; a documentation check never needs network credentials or downloads speech models.

`Examples.cs` wraps each fence in a separate method with the required input parameters and shared namespace imports. Method-only samples are referenced through delegates so the compiler checks them without invoking inference. When updating a fence, update its corresponding method body too; the runtime check detects drift. Model-backed samples compile, but are deliberately not run against arbitrary files or voices by this project. Use the public methods with your own audio and verified models to execute them.
