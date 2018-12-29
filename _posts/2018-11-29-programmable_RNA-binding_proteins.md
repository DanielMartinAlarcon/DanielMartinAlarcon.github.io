---
title: Programmable RNA-binding proteins
subtitle: Primer to a PhD Thesis Project
image: /img/3_pumby/pumby-thumbnail.png
---
![PNAS paper banner](/img/3_pumby/pumby2.png)
![Pumby-RNA structure, with labels](/img/3_pumby/pumby5.png)


Pumby is a programmable protein designed to bind to the secondary genetic messenger, Ribonucleic Acid or RNA, in living mammalian cells. I designed and built Pumby for my PhD thesis while I worked in [Ed Boyden's group at the MIT Media Lab](http://syntheticneurobiology.org/). You can always read the full published paper ([PNAS copy](http://www.pnas.org/content/early/2016/04/25/1519368113); [local copy](http://syntheticneurobiology.org/PDFs/16.04.adamala.FULL.pdf)), but I believe that every good paper deserves an informal recap of what the project was about and why it is interesting. That is what you'll find here.

## Programmable molecules are special

Much of modern bioengineering is based on tools that bind to important biomolecules and execute some useful behavior, preferably in living organisms. The most famous biotechnology of the last decade, [CRISPR](https://en.wikipedia.org/wiki/CRISPR), is based on a protein-RNA complex that can bind to specific sequences of DNA in the genome of basically anything. Many human diseases are caused by small errors in the genome. A protein that can specifically bind to those problematic sequences can also be used to edit them, thereby correcting the problem and eventually leading to a cure. 

One of the great advantages of CRISPR is that it is a programmable complex. Many drugs work by binding to a very specific molecular target somewhere in the body. If the target changes or you want to treat a different disease, you have to design a completely different drug molecule. Programmable drugs, on the other hand, are those where the same molecule can be easily redesigned to bind to completely different targets. We created Pumby because we wanted a programmable molecule that could bind to RNA.

## RNA as a living scaffold
Pumby is a general-purpose tool for [synthetic biology](https://en.wikipedia.org/wiki/Synthetic_biology), but we originally designed it to solve a very specific problem in neuroscience (the Boyden lab is, after all, focused on neurotech). Neurons are very dynamic systems, with lots of internal variables that change over time and affect each other. Neuroscientists frequently monitor these changes using molecules that change their fluorescence in response to some important variable. A calcium sensor like [GCaMP](https://en.wikipedia.org/wiki/GCaMP) shines bright when calcium levels spike, as happens in neurons when they talk to each other. In brains containing GCaMP, then, [you can literally see brain activity in a microscope](https://www.youtube.com/watch?v=FGvp6cdKb3c).

Fluorescent sensors like GCaMP are great, but they have a common problem: the best ones are all the same color. Two different sensors can report simultaneously, but if they're evenly distributed throughout the cell then their signals will overlap and be very difficult to tell apart. I prefer to express this idea in haiku:
 
![Clustering haiku 1](/img/3_pumby/pumby3.png)

What if, however, sensors of a given type clustered together? If you attach sensors to Pumby, and express an RNA molecule with many copies of the sequence that Pumby recognizes, you could build an RNA scaffold that will help the sensors cluster together. Clustered sensors then have two distinct advantages.  First, clusters are way brighter than the same number of sensors spread throughout the whole cell.  Second, clusters of different sensor types are separate from each other and can be monitored independently with the same microscope. And so, Pumby expands the reach of some of the most widely used tools in neuroscience.

![Clustering haiku 2](/img/3_pumby/pumby4.png)

