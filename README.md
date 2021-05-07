This includes my implementation of a "creative adversarial network," or CAN.
As I understand it, the original CAN is similar to a conditional GAN but uses a style ambiguity loss where the generator is trained
to produce images that the discriminator thinks could fall under any class with equal probability. 
My change is to have two training phases. In the first, an InfoGAN is trained so that instead of having human-defined categories of art, 
the network can create its own classes. In the second phase, the q model of the InfoGAN can be frozen so that the network's definition of each class stops changing
and, like in the original CAN, a style ambiguity loss is used.
