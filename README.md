![model image](/figures/manuscript_figures/Figure1.png)

**Code and interactive figures used in *Sensory cortex is optimised for prediction of future input***.

Project website: [https://yossing.github.io/temporal_prediction_model/](https://yossing.github.io/temporal_prediction_model/)

The full eLife paper can be found at: [https://elifesciences.org/articles/31557](https://elifesciences.org/articles/31557)

The interactive supplementary figures can be found [here](https://yossing.github.io/temporal_prediction_model/figures/interactive_supplementary_figures.html)

----

**Code**

All of the custom code used in the project can be found [here](https://github.com/yossing/temporal_prediction_model/tree/master/src)

Detailed instructions on how to use the code can be found in the [README](https://github.com/yossing/temporal_prediction_model/tree/master/src/README.md) 

----

**Abstract**

Neurons in sensory cortex are tuned to diverse features in natural scenes. But what determines which features neurons become selective to? Here we explore the idea that neuronal selectivity is optimised to represent features in the recent sensory past that best predict immediate future inputs. We tested this hypothesis using simple feedforward neural networks, which were trained to predict the next few video or audio frames in clips of natural scenes. The networks developed receptive fields that closely matched those of real cortical neurons in different mammalian species, including the oriented spatial tuning of primary visual cortex, the frequency selectivity of primary auditory cortex and, most notably, their temporal tuning properties. Furthermore, the better a network predicted future inputs the more closely its receptive fields resembled those in the brain. This suggests that sensory processing is optimised to extract those features with the most capacity to predict future input.

----

**Accessible Summary** 

A large part of the brain is devoted to processing sensory input. This processing allows us to tell, for example, if the image we see is of a cat or a dog, or the sound we hear is a bark or a meow. Neurons respond to sensory input by generating spikes of activity. For example, in primary visual cortex, each neuron typically responds best to an edge-like structure moving before the eyes with a particular location, orientation, speed and direction of motion. In primary auditory cortex, each neuron typically responds best to changes in the loudness of sounds over a particular range of sound frequencies.

We sought to understand the neural code used by primary sensory cortex -- why neurons respond to the particular set of stimulus features that they do. For example, why do visual neurons prefer moving oriented edges rather than say rotating hexagons, and why do auditory neurons prefer sounds that change in loudness or frequency composition over time rather than steady unchanging sounds? A dominant hypothesis, which can explain much of the behaviour of sensory neurons, is that neural codes are optimised to be ‘sparse’ -- in other words to minimise the number of spikes required to represent stimuli. We show that a simple alternative principle may explain the code used by the sensory brain -- namely, that neurons use the code that most efficiently allows prediction of future input. This would make sense since features in the world that are predictive of the future will be informative for guiding future actions.

To do this, we simulated networks of neurons in a computer. We optimised the connection strengths of these neurons so that they efficiently predicted the immediate future of videos of natural scenes from their past. We then examined the preferred stimuli of the simulated neurons. These turned out to be moving, oriented edges, just as in the real primary visual cortex of mammals. We also optimised the same simulated network to predict the immediate future of recordings of natural sounds from their past.  The resulting auditory stimuli preferred by the neurons also closely matched those preferred by neurons in the real primary auditory cortex. In particular, for both vision and audition, the temporal structure of these preferred stimuli was similar to that found for real neurons -- which is not the case for other principled models such as sparse coding.

![model image](/figures/manuscript_figures/Figure2.png)

Our results suggest that coding for efficient prediction of the future may be a general principle behind the way the brain represents the sensory world. Disorders of sensory processing are unfortunately all too common, and a better understanding of the computational principles underlying sensory processing should help us to interpret what goes wrong in the brain and why. Temporal prediction may also be relevant to machine learning and artificial intelligence applications, providing a simple method by which smart devices might be trained to process sensory inputs.
