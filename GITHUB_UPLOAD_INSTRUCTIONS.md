# GitHub Upload Instructions

## Browser method

1. Create a new public repository named `persian-absa-llama3`.
2. Do not initialize it with another README or license.
3. Extract this ZIP file.
4. Upload the **contents** of the `persian-absa-llama3` folder, not the outer folder itself.
5. Confirm that notebooks contain no outputs, tokens, local paths containing personal credentials, or private datasets.
6. Publish the repository.
7. Copy the public URL and replace the placeholder in the manuscript:

```text
https://github.com/USERNAME/persian-absa-llama3
```

No annotation or dataset files should be added unless their redistribution rights and consent conditions have been separately verified.

## Git command method

```bash
git init
git add .
git commit -m "Initial reproducibility release"
git branch -M main
git remote add origin https://github.com/USERNAME/persian-absa-llama3.git
git push -u origin main
```
