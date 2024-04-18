# ocean_emulators

## Developing this package

Set up a fresh mamba environment (Optional but recommended)

```bash
mamba create -n -n=ocean_emulators_dev python=3.10
mamba activate ocean_emulators_dev
```

>[!NOTE]
> If you are using conda instead of mamba, you can replace `mamba` with `conda` above

Now install this package with all the developer extra dependencies

```
pip install -e ".[dev]"
```

Before you edit the code make sure all tests pass by running pytest from the root level of this repository
```
pytest
```
