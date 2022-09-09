# From detectors to publications

How do we gain knowledge about the Universe in experimental particle physics?

This interrogation intrigued me so much that it got me to leave engineering and start a Ph.D. to answer the question.

```{note}
Experimental particle physics is the __context__ of this machine learning course, from which examples will be taken.

This section is not a physics lecture, rather an introductory tour in the subatomic world. It aims at:
* expanding your knowledge in a new area of mathematical sciences
* showing how much maths and computing is behind the discipline 
* presenting the opportunities to work in the field 
```

## What is particle physics?
The goal of physics is to understand the universe. Quite a task. Among the numerous branches of physics, particle physics focuses on the tiniest chuncks of matter: elementary particles. 
It describes how the known elementary particles interact through three of the four fundamental forces, or interactions: electromagnetism, weak interaction and strong interactions. 
What about gravity? The great Albert Einstein tried to include it until his death without success; merging the theories of Quantum Mechanics and General Relativity is still one of the hardest physics problem today.

So you got the spoiler alert: the theory of particle physics, bearing the weird name of "Standard Model," is not complete as it does not include all fundamental interactions.
Despite this important caveat, the model is a triumphant achievement made by many great physicists since the 1950s (with a large fraction becoming Physics Nobel Prize laureates).
Not only the theoretial framework accurately describes the known subatomic world, it has accumulated remarkable successes with numerous predictions verified in experiments at an incredible level of precision.

### Theoretical vs experimental particle physics
In all physical sciences, knowledge is forged via the process known as the scientific method.

```{admonition} In-class exercise
:class: dropdown
Take 5 min to define the scientific method. Write full sentences or keywords in a bullet list.
Then compare with your neighbours your definition.
What notion(s) did you put that they omitted? What did you miss?
Form a merged definition with two or three classmates.
```

In particle physics there has always been a back-and-forth between theory and experiments.
Either a new particle is discovered and no one understand why. It was the case in 1936 with the discovery of the muon, a heavier cousin of the electron. It surprised the
community so much that Nobel laureate Isaac Rabi famously quipped: "Who ordered that?" The other way around is more common: the model provides a prediction, usually in the form of a new particle or a new process (particle interaction) that can be observed in the data from particle detectors. The theory of particle physics has drawn its succcess from the numerous experimental confirmations of theoretical predictions, with mind-blowing precision at the part-per-billion level. 

### State-of-the-art machines 
Often a conceptualized new particle would require several decades for the experimental setup to be ready, as its observation demands a higher energy beam, a finer resolution or both. In that sense, particle physicists works hand-in-hand with engineers towards pushing the limits of technology. The quest of one particle is remarkable; the Higgs boson was a missing piece in the Standard Model, responsible of giving to other particles their respective masses. It took 48 years between its formulation in 1964 and the discovery in 2012 at the [Large Hadron Collider (LHC)](https://home.cern/science/accelerators/large-hadron-collider) at the European Organization for Nuclear Research near Geneva, Switzerland. The LHC is one of the most complex machines ever built by human beings and collects several [superlatives](https://home.cern/resources/faqs/facts-and-figures-about-lhc). I happen to work on another endeavour, currently under construction, called the [Deep Underground Neutrino Experiment (DUNE)](https://www.dunescience.org/). It will demands a newer generation of detectors due to the size and the intensity of the incoming beam. Unique challenges lie ahead for this future biggest neutrino detector in the world, especially in computing. Due to enormous physical data volumes that need to be acquired, stored and analyzed, the requirements of DUNE is likely to trigger a paradigm shift with groundbreaking new techniques in data science.

```{figure} ../images/lec01_ATLAS_legend_144-dpi.jpg
---
width: 800px
name: ATLAS Detector
---
. Schematic of the ATLAS detector with its subsystem. People (who are not allowed to climb by the way) are drawn for scale. ATLAS is the largest volume detector ever constructed for a particle collider. You can take a [virtual tour](https://atlas.cern/Discover/Detector) here. Credits: ATLAS Experiment © 2021 CERN.
```

### Many activities, many people 
Particle physics is a discipline offering a wide range of sub-activities, in particular on the experimental side. On the data analysis side, the numerous steps and various associated tasks will be covered in the coming section {ref}`trailer:hep:howAna`. It is also possible to contribute in hardware projects, such as test beam campaigns and detector upgrade programs. Getting to design, test or build the next generation of particle detectors can be very exciting! Many simulations are required, so even on this hardware and engineering related side, mathematicians and programmers can help. At the interface between hardware and software lies data acquisition. Experts here (engineers and also physicists) ensure the microelectronics are proper to deliver quality data in due time. If you think this is not where machine learning techniques would operate, hold on: very recent multi-disciplinary proposals are eager to implement machine learning algorithms on programmable hardware for pattern recognition! This would enable pattern recognition and particle identification to operate 'live', i.e. during data taking. Many particle physics experiments are eager to implement these fast techniques in order to know as soon as possible if the fresh data contains interesting physics or not.

```{admonition} Learn more
:class: dropdown
A programmable hardware is an integrated circuit with configurable logic blocks that can be wired together using a special software. The most popular programmable logic device is the field-programmable gate array (FPGA). It is widely used in particle physics experiments as well as in other electronics applications. Yet programming machine learning algorithms on FPGAs (labelled ML-FPGA) is a new effort that, due to the requirements in particle physics, is very promising.

Further readings on hardware acceleration

* __General:__ [Of hardware acceleration in machine learning](https://medium.com/unpackai/of-hardware-acceleration-in-machine-learning-38b9726199eb)
* __Specific to particle physics:__ (and a bit technical) [Particle identification and tracking in real time using Machine Learning on FPGA](https://www.jlab.org/sites/default/files/eic_rd_prgm/files/2022_Proposals/ML_FPGA_R_D_FY23proposal_v2_EICGENRandD2022_15.pdf).
```

Nowadays, most particle physics endeavours can not be achieved alone. The magnitude of the experiments, their complexity and the resulting workload require a highly collaborative and international environment. For instance the ATLAS Collaboration, associated with the largest general-purpose particle detector experiment at the Large Hadron Collider (LHC), comprises over 5900 physicists, engineers, technicians, students and administrators. ATLAS has 2900 scientific authors from over 180 institutions. It is one of the largest collaborative efforts ever attempted in science. The more recent DUNE Collaboration has already 1400 members and is growing.

### Spin-offs
In the quest to better understand the universe, particle physics has created by-products and even new disciplines, some drastically changing our lives. One striking example is the World Wide Web, invented at CERN by computer scientist Tim Berners-Lee in 1989. At the start, the so-called HTTP protocol and first web server were a mean to manage documentation. A couple of years later, this seamless network that any computer would be able to access revolutionized the way information was shared and the way we communicate, socialize, and conduct business.

Particle physics brought several breakthrough technology in medical physics, a branch of applied physics which has emerged since the discovery of x-rays by Wilhelm Röntgen in 1895. In the 1950s a detector called Position Emission Tomography (PET) was used to visualize inside the body, from metabolic processes to tumorous cells. Years later, driven by the technical challenges posed by the Large Hadron Collider (LHC) at CERN, innovative material and chip design used in state-of-the-art LHC detectors were implemented in PET prototypes to increase their resolution. This was imaging for diagnosis. Another technology-transfer arising from the development of linear accelerators was radiotherapy: the beam of accelerated particles is directed into the patient's body to kill tumor tissue. 

```{admonition} Learn more
:class: dropdown
Articles about CERN's activities benefiting medical physics: 
* [CERN’s impact on medical technology, CERN Kownledge Transfer group](https://kt.cern/news/opinion/knowledge-sharing/cerns-impact-medical-technology)
* [How the LHC could help us peek inside the human brain](https://home.cern/news/news/knowledge-sharing/how-lhc-could-help-us-peek-inside-human-brain)
```

(trailer:hep:howAna)=
## How do we analyse data?
Back to our original question.

Although the data and physics outcomes are different between the various particle detectors, there is a common series of steps shared in data analyses.

The raw data from particle detectors is a (large) collection of activated electronic channels from the detector's readout material. Such readout material can be an array of wires, or sensitive plates, usually in a high number to cover a large area or volume, similar to tiles covering a rooftop. 

```{figure} ../images/lec01_CMS_tracker.jpg
---
width: 80%
name: CMS Tracker
---
  
. The tracker of the [CMS experiment](https://cms.cern/detector), one of the detectors of the LHC ring.  
Credits: CMS Experiment © 2021 CERN.
```
```{figure} ../images/lec01_ATLAS_SCT.jpg
---
width: 80%
name: ATLAS SCT
---
. Workers assembling the ATLAS SemiConductor Tracker (SCT) at CERN.  
Credits: ATLAS Experiment © 2021 CERN.
```

### 1. The trigger
When particles with sufficient energy pass through the wires or plates, a current is produced and picked up by the electronics. If several adjacent wires or plates are activated at the same time, chances are, an interesting particle interaction has taken place.
The first step at the start of the data acquisition is the trigger: a combination of hardware and software selects the most interesting interactions for study.

A particle detector is analogous to a camera: it takes 'pictures' of interactions of interest. Two important points:
* The interactions of interest are usually not alone. Either there is a mess of other particles produced - that is the case in colliders with man-made energetic collisions. Alternatively, detector expecting rare signals can have impurities in them causing noisy interactions mimicking the ones physicists are looking for. We refer to these unwanted events as background, opposed to the signal, the interactions we want to record and analyse later. 
* The pictures are not common pictures, they need post-processing, i.e. the 3D interaction needs to be 'reconstructed.' 

A recorded interaction is labelled an 'event.' It is an undeveloped photography. As we don't know yet which particles are at play, we collect a very large amount of events that will be pre-processed and later analyzed statistically.

### 2. Event reconstruction
As stated above, a raw event contains all the triggered electronics from the numerous wires or planes of all sub-detector systems. To put it in mathematical terms: it is a large collection of dots, with their coordinates and a timestamp. It is impossible to visualize as-is nor start any data analysis yet. The reconstruction step is necessary to create more visual entities such as tracks representing the particle trajectories. 

```{figure} ../images/lec01_tracks_Andreas.png
---
width: 100%
name: ATLAS tracks
---
. Front and side view of the cylindrical ATLAS inner detector with recorded hits (dots) and candidate tracks (blue). Credit: Andreas Salzburger.
```

Algorithms are given as input all dots coordinates and work out the best combination to draw tracks connecting them. At the end, the data from a given event contains a bunch of tracks, which can be straight or curved, vertices, which are intersections between tracks, and the amount of 'deposited energy', that is to say the energy the particle put into a special readout material while slowing down, giving us access to its initial energy (crucial for the particle's identification).

### 3. Particle identification
With the tracks, vertices and deposited energy information, it is possible to identify the different particles that were present in that given event, with even their initial speed and direction right after the interaction in which they were produced. Many different algorithms are employed at this stage, all specific to the particle they aim at identifying. A lot of those algorithms use machine learning techniques. At the end of the particle identification step, the information we have can be illustrated as a 3D rendering (you can see an example [here](https://twiki.cern.ch/twiki/pub/AtlasPublic/EventDisplayRun2Physics/FourTopsEvent.png) I can detail to you if you are curious).

### 4. The comparison
At this stage we have a picture of the interaction's detected 'objects', i.e. the identified particles, their energy and direction from the moment they were produced. 
But these objects are usually the secondary products of the interaction of interest. Most key particles are decaying shortly after being produced without even reaching the detector's readout material. Consequence? With a single picture of the detected products of an interaction, we can not know which initial particles were present. Moreover, known processes are often producing the same secondary products. The only way we can know is statistical, by collecting many of these and seeing the trends between the signal (the process we want as predicted by theorists) and the known mimicking processes we don't know, aka the background (sometimes referred as noise in other fields). We know the signal and background trends using simulated samples: it looks like the data but the interactions were generated by dedicated algorithms. 

In both real and simulated data, we compute special entities, quantifying the topology of the recorded objects from the interaction. For instance, it can be the norm of the vectorial sum of two particles' speed vector. To visualize the trends, we plot the data as histograms to see how the variable is distributed in a given range of values for both signal and background. After numerous studies with more variables, checks and more checks, we can overlay two distributions: 
* the background and signal distributions, indicating the number of predicted events (vertical axis) for a given range of the plotted variable (horizontal axis)
* the real data - remember, we don't know what's inside!
````{margin}
The technical details of this plot are way beyond the scope of this course. Yet I want to illustrate the points above and show you a real plot that is part of the Higgs boson discovery paper! If you are curious, I am happy to explain more and share extra reading (for now this is an [excellent one](https://home.cern/science/physics/higgs-boson)).
````
```{figure} ../images/lec01_H4lep.png
---
  name: Higgs_boson_distribution
  width: 80%
---
 . Overlaid distributions of the data (black dots) with the simulated data (colored filled histograms). The predicted signal process is the Higgs boson in blue, while background processes (red, purple and yellow). The variable $m_{4l}$ is analogous to a mass. There are two resonances (we call them mass peaks) and you can see that the data points overlay well with the predicted Higgs peak. Credits: ATLAS Collaboration/CERN. 
```


### 5. The test statistics



### 6. The conclusion



And this ends up in the updated version of physics textbook, testimony of the accretion of knowledge.


## The maths and computing in particle physics
Although the field belongs to physics, a lot of mathematic and computing lie in the exploration of the subatomic world. During this tour I will briefly mention the concepts and show the connections between what you may have learned before in pure mathematics and their use in the reality we live in.

### Symmetries, groups and 
_The Standard Model is a quantum field theory that is based on a gauge symmetry._  
Let's unwrap this sentence.





_Lagrangian - principle of least action - symmetries as 'invariance' Noether theorem - gauge theory - abelian groups_
_The universe we live in is a particular of gauge symmetry_ I find it mind-blowing that not-only reality 

### Phystatisticians 



### Monte Carlo generators
In HEP, sophisticated simulation programmes are used to design the experimental set-ups and to interpret the data.


in short 'Montecarlos' - 

https://www.symmetrymagazine.org/article/the-coevolution-of-particle-physics-and-computing?language=en


## Today's landscape in particle physics

### On the theory side
where experimental data clashes with the current model. 



### On the experimental side
Not exhaustive but limited to my areas ... 



### Opportunities for mathematicians and programmers



