# Computer Vision Aided Beam Tracking in A Real-World Millimeter Wave Deployment

This is a Python code package related to the following article:
S. Jiang and A. Alkhateeb, "[Computer Vision Aided Beam Tracking in A Real-World Millimeter Wave Deployment](https://ieeexplore.ieee.org/document/10008648)," in IEEE Globecom Workshops, 2022

# Abstract of the Article
Millimeter-wave (mmWave) and terahertz (THz) communications require beamforming to acquire adequate receive signal-to-noise ratio (SNR). To find t he o ptimal beam, current beam management solutions perform beam training over a large number of beams in pre-defined c odebooks. The beam training overhead increases the access latency and can become infeasible for high-mobility applications. To reduce or even eliminate this beam training overhead, we propose to utilize the visual data, captured for example by cameras at the base stations, to guide the beam tracking/refining process. We propose a machine learning (ML) framework, based on an encoderdecoder architecture, that can predict the future beams using the previously obtained visual sensing information. Our proposed approach is evaluated on a large-scale real-world dataset, where it achieves an accuracy of 64.47% (and a normalized receive power of 97.66%) in predicting the future beam. This is achieved while requiring less than 1% of the beam training overhead of a corresponding baseline solution that uses a sequence of previous beams to predict the future one. This high performance and low overhead obtained on the real-world dataset demonstrate the potential of the proposed vision-aided beam tracking approach in real-world applications.
# License and Referencing
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). 
If you in any way use this code for research that results in publications, please cite our original article:
> S. Jiang and A. Alkhateeb, "Computer vision aided beam tracking in a real-world millimeter wave deployment," in IEEE Globecom Workshops, 2022.

If you use the [DeepSense 6G dataset](www.deepsense6g.net), please also cite our dataset article:
> A. Alkhateeb, G. Charan, T. Osman, A. Hredzak, and N. Srinivas, “DeepSense 6G: large-scale real-world multi-modal sensing and communication datasets,” to be available on arXiv, 2022. [Online]. Available: https://www.DeepSense6G.net
