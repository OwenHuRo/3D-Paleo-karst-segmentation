# Third-party code and research foundations

Parts of the transformer baseline were adapted from the official UCTransNet
implementation:

- Repository: <https://github.com/McGregorWwww/UCTransNet>
- Paper: H. Wang, P. Cao, J. Wang, and O. R. Zaiane, "UCTransNet: Rethinking
  the Skip Connections in U-Net from a Channel-wise Perspective with
  Transformer," AAAI 2022.
- Relevant files include `Models/UCTransNet.py`, `Models/CTrans.py`, the
  center-position transformer variants, and `Utils/Config.py`.

The U-Net and U-Net++ baselines build on their respective published
architectures. Their papers are cited in the accompanying manuscript.

The repository-level MIT license applies to original project code. Copyright
and any applicable terms for third-party-derived portions remain with their
original authors. Before redistributing those portions, the maintainer should
confirm and record the applicable upstream permission or license. This notice
does not replace upstream license terms.
