### Using `TransformerLens` with DistilBERT

Due to limitations in the current implementation of the `TransformerLens` library, we are unable to use DistilBERT models.

##### 1. Add `TransformerLens` as a submodule
```bash
git submodule add https://github.com/TransformerLensOrg/TransformerLens external/TransformerLens
git submodule update --init --recursive
```

##### 2. Create custom branch
```bash
cd external/TransformerLens
git checkout -b add-distilbert-support
```

##### 3. Point main project to submodule
```bash
pip install -e external/TransformerLens
```

##### 4. Commit submodule reference
```bash
git add .gitmodules external/TransformerLens
git commit -m "."
```

##### 5. Clone with submodules (if required)
```bash
git clone --recurse-submodules git@github.com:themarcosf/research.git
# or, if already cloned
git submodule update --init --recursive
```

##### 6. Fetch changes from upstream
```bash
cd external/TransformerLens
git remote add upstream https://github.com/TransformerLensOrg/TransformerLens
git fetch upstream
git merge upstream/main  # resolve conflicts if needed
```