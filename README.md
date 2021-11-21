# Robust-Time-Difference-of-Arrival-TDOA-Estimation
This work try to reimplement "Robust TDOA Estimation Based on Time-Frequency Masking and Deep Neural Networks".
In stage1, using rir generator from https://www.audiolabs-erlangen.de/fau/professor/habets/software/rir-generator to create sample data.
In stage2, modifing open-unmix model to predict masks.
In stage3, using predicted mask and steering vector estimation to predict direction of arrival in noisy condition.
