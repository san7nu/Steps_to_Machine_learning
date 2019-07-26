# Steps to Machine learning


##### `Disclaimer `
This is my path of learning. 

Each statement here is near approximation to real ones. As learning directly the real-ones always involves too much depth and i get bored too easily. So i am going by thought that, we need that much knowledge which is neccessary to move to next step and by the time we reach our target through small corrections we will attain most of needed knowledge.

---
### Table of Contents  
[Intro](#intro)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[End-goal of Artificial Intelligence(AI)](#endgoal)<br /> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Scope](#scope)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Machine Learning](#ml)<br />
[Types of Learning](#types)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Supervised](#supervised)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Unsupervised](#unsupervised)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Discriminative](#disc)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Generative](#genr)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Bayesian Networks](#bayes)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Time-series analysis](#times)<br />
[ML Algos](#algos)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Nearest Neighbor, KNN](#knn)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Linear regression](#lireg)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Logistic regression](#loreg)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[SVM(Support Vector Machine)](#svm)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Random Forest](#randf)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Naive Bayes](#nbayes)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Dimentionality reduction(PCA, LDA)](#pca)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Neural Networks](#nn)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Convolutional Neural Networks](#cnn)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Recurrent Neural Networks](#rnn)<br />
[Libraries](#lib)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Keras](#keras)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Tensorflow](#tf)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Torch](#torch)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Theano](#theano)<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Caffe](#caffe)<br />
[Recent research](#rsrch)<br />

----
<a name="intro"/>

### Intro
This intro can be [skipped](#types), but show little patience and read it can definetely add directional sense to the overall path/step.  

Machine learning world assumes that there are only two problems in this whole universe
1. Prediction/Regression problem: Predict something based on past
2. Classification problem: Classify something based on past

Every problem can be directly or indirectly made into these two problems.

<a name="endgoal"/>

##### `End-goal of Artificial Intelligence(AI)`
To create "something" which can make machines mimic human behavior.

<a name="scope"/>

##### `Scope`
What does it mean by mimicing human behaviour ?.. If machine can:

Reason, Represent knowledge, Plan, Learn, Process natural language(talk & listen), Perceive, Motion and Manipulation, Extra(Social & general) intelligence

There are many techiques which can make machines do these, one of the most powerful one is "Machine Learning"

<a name="ml"/>

##### `Machine Learning`

Idea is that there are generic algorithms that can tell something interesting about a set of data without we having to write any custom code specific to the problem.

In simple words, if we show data to code then it itself finds pattern inside the data(depending on data).
In further simple words, if i show images to code then it itself finds pattern in them and recognizes then dog, cat,....(too ambitious i know :)

Let take an example in which we have two inputs <br />
Example1:<br />
Input1 = 1,2,3,4,5,6,7,8,9,10,11........100<br />
Input2 = 0,1,0,1,0,1,0,1,0,1,0...........1

and our task is to write some code that takes these two inputs and in future if someone gives it Input1(lets say 2312), then it has to find value of Input2(to be 1).

Yes its even-odd program. 

Example2:<br />
Input1 = 1,2,3,4,5,6,7,8,9,10,11........,99,100<br />
Input2 = 0,0,1,0,0,1,0,0,1,0,0...........1,0

and our task is to write some code that takes these two inputs and in future if someone gives it Input1(lets say 999), then it has to find value of Input2(to be 1).

Yes its 3-divisbles.

Example3:<br />
Input1 = 1,2,3,4,5,6,7,8,9,10,11........,99,100<br />
Input2 = 0,0,0,0,1,0,0,0,0,1,0...........0,1

and our task is to write some code that takes these two inputs and in future if someone gives it Input1(lets say 9850), then it has to find value of Input2(to be 1).

Yes its 5-divisbles.

Now question is, can we write a generic code which can work for either of above example(1,2,3) and can handle anything of such kind ?<br />
May be you figured out something like this, you track the 1st True(1) in Input2 and find its divisibles.<br /> 
In this you have given your code `Intelligence` to learn itself which number to divide. This is kind-of basic idea of AI, where we find some algo which has intelligence inside itself rather than we solving each task with specific logic, we try solve all the scenarios in the world with single logic (again this too optimistic target, in reality we use small bunch of algos to solve all problems)

---
<a name="types"/>

### Types of Learning
<a name="supervised"/>

##### `Supervised Learning`
I show to code some inputs and outputs initially(this phase is called training) and later(this phase is called testing) i will ask what are the outputs for given inputs.<br />
For example, i will show hell lot of dog and cat pictures and tell which is dog and cat then ask our code at the end to identify a dog or cat based on new picture i show.

<a name="unsupervised"/>

##### `Unsupervised Learning`
I don't tell anything about output, i will give lot of data and i expect code to figure out some relationships or segregations in data.<br />
For example, i throw lot of dogs and cat pictures, i expect code to find `segregation` of what is dog and cat.<br />

It differs from supervised learning in a way that we don't tell what is dog and cat, we expect code to somehow figure out that there are two different objects.
<a name="disc"/>

##### `Discriminative Learning`
Estimate p(y|x) probability of y given x.
<a name="genr"/>

##### `Generative Learning`
Estimate p(x,y) or p(x|y).p(y) combined probability of x and y
<a name="bayes"/>

##### `Bayesian network`

* Bayes Theorem
* Casual BN
* Pystan
<a name="times"/>

##### `Time-series analysis`
* AR(Auto-regression)
* MA(Moving average)
* ARMA(AR+MA)
* ARIMA(AR+ Integrated +MA)
* SARIMA(Seasonal ARIMA)
* Exponential smoothening
* Holt-winters
* VAR(Vector AR)
* Auto-ARIMA
* Prophet
<a name="algos"/>

### ML Algos
