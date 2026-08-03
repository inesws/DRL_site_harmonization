# DRL_site_harmonization
This is the official code repository for "Disentanglement learning to deconfound neuroimaging data: application to multi-site data harmonization in psychiatry" (in update).

# Scripts:
FC_ENV_fusion_xcov_at_zfusion.py: fusion of brain functional connectivity and environmental data, with xcov site-disentanglement at fusion bottleneck OR site and sex disentanglement: variation 2 step or joint optimisation;

FC_ENV_fusion_xcov_at_zFC.py: fusion of brain functional connectivity and environmental data, with xcov site-disentanglement at FC embeddings OR site and sex disentanglement : variation 2 step or joint optimisation;

FC_ENV_fusion_COMBAT.py: fusion of brain functional connectivity and environmental data, with multi-site harmonisation of FC with ComBat;

# Requirements
python = 3.9.18
conda create -n myenv python=3.9.18
pip install -r requirements.txt

# For combat function check:
https://github.com/inesws/neurocombat_pyClasse.git

# Ref:
I. W. Sampaio et al., "Disentanglement Learning to Deconfound Neuroimaging-Environmental Data: Application to Multi-Site Data Harmonization in Psychiatry," in IEEE Access, vol. 14, pp. 111815-111827, 2026, doi: 10.1109/ACCESS.2026.3712218.



