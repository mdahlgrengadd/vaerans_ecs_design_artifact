# Google Colab Setup Guide

## Quick Start

1. **Upload the notebook to your GitHub repository**
   - Ensure `tutorial.ipynb` is in the `examples/` directory
   - Update the repository URL in the notebook

2. **Update GitHub repository URL**
   - Open `tutorial.ipynb` in a text editor
   - Find and replace `YOUR_USERNAME/vaerans_ecs_design_artifact` with your actual GitHub username and repo name
   - Update in two places:
     - The Colab badge link at the top
     - The `repo_url` variable in the setup cell

3. **Open in Google Colab**
   - Go to [https://colab.research.google.com](https://colab.research.google.com)
   - Click "File" → "Open notebook"
   - Select "GitHub" tab
   - Enter your repository URL or username
   - Select `examples/tutorial.ipynb`

   **OR**
   
   - Click the "Open in Colab" badge at the top of the notebook (after updating the URL)

4. **Run all cells**
   - Click "Runtime" → "Run all"
   - The notebook will automatically:
     - Clone your repository
     - Install the `vaerans_ecs` package
     - Install dependencies (PIL, matplotlib)
     - Run all examples

## What Gets Installed

The setup cells automatically install:
- `vaerans_ecs` package (from your GitHub repo)
- `numpy` (dependency)
- `onnxruntime` (for VAE models)
- `pywt` (wavelets)
- `constriction` (ANS entropy coding)
- `scikit-image` (metrics)
- `pillow` (image I/O)
- `matplotlib` (visualization)

## Example GitHub URLs

If your repo is at `https://github.com/johndoe/vaerans-ecs`:
```python
repo_url = "https://github.com/johndoe/vaerans-ecs.git"
```

Badge link:
```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/johndoe/vaerans-ecs/blob/main/examples/tutorial.ipynb)
```

## Private Repositories

For private repositories:
1. Generate a GitHub Personal Access Token:
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Create token with `repo` scope
2. Update the clone command in the notebook:
```python
repo_url = "https://<TOKEN>@github.com/<USERNAME>/<REPO>.git"
```

## Troubleshooting

### "Repository not found"
- Verify the repository URL is correct
- Ensure the repository is public, or use a personal access token

### "Module not found"
- Ensure the setup cell completed successfully
- Restart runtime: "Runtime" → "Restart runtime"
- Run setup cells again

### ONNX model errors
- The notebook uses test fixtures from `tests/fixtures/`
- For production models, update `vaerans_ecs.toml` with correct paths
- You can upload model files to Colab using the file browser

## Custom Configuration

To use custom ONNX models on Colab:
1. Upload your ONNX files using the Colab file browser
2. Create/modify `vaerans_ecs.toml`:
```python
from pathlib import Path
config_content = """
[models.sdxl-vae]
encoder = "/content/vae_encoder.onnx"
decoder = "/content/vae_decoder.onnx"
"""
Path("vaerans_ecs.toml").write_text(config_content)
```

## Performance Notes

- **GPU**: Colab provides GPU access (Runtime → Change runtime type → GPU)
- **Memory**: Free tier has ~12GB RAM, sufficient for most examples
- **Session Time**: Free tier sessions timeout after inactivity
- **Persistence**: Files in `/content/` are deleted when session ends

## Support

- 📖 Documentation: See `SOFTWARE_DESIGN.md` in the repository
- 🐛 Issues: Report at GitHub Issues
- 💬 Questions: Open a discussion on GitHub
