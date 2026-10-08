# Licensing and Upstream Code

This repository contains original LR-V2X contributions and code adapted from
other research projects. It is not licensed as a whole under MIT.

## Original Contributions

[LICENSE-LR-V2X](LICENSE-LR-V2X) applies to the original README, citation metadata,
configuration additions, and the independently implemented latent encoder and
latent prior decoder (`latent_encoder.py` and `simple_prior_decoder.py`).
It does not override any upstream terms attached to copied or derived code.

## OpenCOOD and HEAL

The detection framework, data loaders, fusion baselines, training utilities,
and their modifications retain their upstream terms and attribution headers.
The inherited [Academic Software License](LICENSE) is preserved. Some files
also carry `TDG-Attribution-NonCommercial-NoDistrib` notices.

- OpenCOOD: https://github.com/DerrickXuNu/OpenCOOD
- HEAL: https://github.com/yifanlu0227/HEAL

These terms require permission for redistribution. Keep this staging repository
private until redistribution permissions or an appropriate upstream replacement
are in place. Adding original MIT-licensed files does not remove that requirement.

## DiT

`opencood/models/sub_modules/diffusion_model_dit.py` adapts the official DiT
implementation. Its upstream license is CC BY-NC 4.0, not MIT.

- Source: https://github.com/facebookresearch/DiT
- License: [licenses/DiT-CC-BY-NC-4.0.txt](licenses/DiT-CC-BY-NC-4.0.txt)

## Other Dependencies

Installed dependencies retain their own licenses. Dataset files and pretrained
weights are not included. Obtain datasets from their official providers and
follow their usage terms.
