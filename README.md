### Overview

This project serves to optimize United States congressional district maps in order to achieve improved outcomes in representation metrics and, hopefully, congressional productivity (i.e. more and better laws passed). The program achieves this by employing a genetic machine learning algorithm that gradually optimizes a population of district maps through random heuristic mutation and fitness selection, evaluating the maps on several important metrics, including population balance, compactness, win margin, and efficiency gap. In the future, I hope to try to bring the project and algorithm to the attention of political organizations or figures to attempt to make an actual impact on the current congressional redistricting system.

So what are the issues with congressional districting exactly? The obvious one is Gerrymandering, which this project works to solve by optimzing maps to be fair to both parties (lower efficiency gap). However, the one that inspired this project is the lack of competition in congressional elections, stemming from uncompetitive congressional districts. The lack of competition is bad because it causes the election of extreme representatives, inhibiting bipartisan collaboration and decreasing congressional productivity, which is why the current U.S. congress has passed so few laws and seems to be unable to collaborate accross parties. This program  optimizes maps to increase competition in the congressional districts (lower win margin) in hope to combat this issue.

The project is organized in a way that the code is split into several files, the purpose of which is to keep the project more organized due to the large amount of code. The program starts in the "main.py" file, which imports other files and begins the main algorithm. Since it may be difficult to obtain a sense of the algorithm through reading through the code (as the codebase is large), please view the pseudocode for a high-level overview that details the necessary knowledge to understand the algorithm.


### Results

The project has so far been an huge success. When I began, I did not expect it to achive this good of results. However, it is now able to beat the current maps in both Pennsylvania and North Carolina (the only two states on which I have tested) by significant margins in almost every metric, showing huge promise to be applicable and beneficial to all states in the nation.

For example, here are the comparison stats that the program achieves for North Carolina:

Current District Metrics:
    Fitness: -0.2808
    Contiguity: 100.0000%
    Population Balance: 6.3153%
    Compactness: 33.5974%
    Win Margin: 20.2973%
    Efficiency Gap: 9.8053%
    
New Solution Metrics: 
    Fitness: 0.1238
    Contiguity: 100.0000%
    Population Balance: 0.2886%
    Compactness: 34.7663%
    Win Margin: 14.1059%
    Efficiency Gap: 6.8380%

In this comparison, the new North Carolina solution has a higher overall fitness, equal contiguity, significantly lower population balance, higher compactness, significantly lower win margin, and significantly lower efficiency gap, all of which are very positive signs.


### Instructions

To begin, go to the project's hosting URL on replit.com and hit run. If necessary, change the user interface input variables, such as which state you would like to optimize or some of the algorithm parameters. Then, begin the algorithm from the user interface. 

The algorithm should now start by loading the voter block data and the pre-generated random maps to fill the population. It may generate more maps if necessary, which can sometimes take a while. After the population is filled, it will begin optimizing the map. 

To achieve noticeable results, running for at least a few hours is necessary, so please remain patient as the program continues and come back to the interface to see results. In the interface, you can compare the current optimized solution found by the algorithm to the map currently used in the state, as well as the fitness and metric stats for the two maps.

To just view the images of current and solution maps, go to the "maps" directory, then the "images" directory, and click on the image files that you want to see.
