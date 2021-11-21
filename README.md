# Robust-Time-Difference-of-Arrival-TDOA-Estimation
This work try to reimplement "Robust TDOA Estimation Based on Time-Frequency Masking and Deep Neural Networks"[[1]](#1).
In stage1, using rir generator from https://www.audiolabs-erlangen.de/fau/professor/habets/software/rir-generator to create sample data.
In stage2, modifing open-unmix model to predict masks.
In stage3, using predicted mask and steering vector estimation to predict direction of arrival in noisy condition.

Following are the results. 'Paper_IRM' and 'Paper_mask' are the results from the reference paper.
![alt text](https://github.com/yihliang831209/Robust-Time-Difference-of-Arrival-TDOA-Estimation/blob/main/image/results.PNG?raw=true)


## References
<a id="1">[1]</a> 
Wang, Zhong-Qiu, Xueliang Zhang, and DeLiang Wang. "Robust TDOA Estimation Based on Time-Frequency Masking and Deep Neural Networks." Interspeech. 2018.
