<a name="br1"></a> 

## Description
We propose a cognitive model that accounts for participant sentiment and facial

expression in the context of the British game show “Split or Steal” (later named “Golden Balls”)

to determine whether an individual would either behave cooperatively or competitively in the

game. To understand the logic behind the model, it is first necessary to comprehend the game’s

rules. In “Split or Steal”, two players are presented with a high sum of money and have one of

two options: either “splitting” or “stealing” the money. If both players decide to split, the two of

them exit the game with half of the total value. On the other hand, if one chooses to split and the

other to steal, the one who chose to steal walks away with the entire prize value. Lastly, if both of

them steal, neither gets any reward (Van Den Assem, 2012). Clearly, in the scenario just

described, it is appropriate to consider game theory-based approaches to modeling human

behavior in order to accurately predict participant decisions.

However, game theory alone isn’t sufficient to account for all the factors that may

influence decision-making, such as past experiences, mood fluctuations, and financial struggle.

Rather, it assumes that participants will attempt to avoid the worse-case scenario, which in this

context means losing all of the money to the other player. Thus, from a game-theory standpoint,

participants should always behave cooperatively in order to ensure some form of financial

compensation for both parties (Mérő, 1998). Unfortunately, such expected behavior is not always

observed, revealing the need to incorporate additional data points to any algorithm that can

accurately predict participant choices in the game. We believe that one such data point is facial

expressions, which can provide valuable insights not only about participant emotion (“sad”,

“happy”, “surprised”, etc.) but also sentiment (“positive”, “negative”, “neutral”). Together, these

two metrics constitute valuable indicators of what a participant is reasoning during the game, and

were thus incorporated into the cognitive model that our group constructed. To implement our



<a name="br2"></a> 

approach, we divided our model into four steps: image recognition, sentiment analysis, game

theory-based decision making, and the production of a final output (“split” or “steal”).

During the image recognition phase, our model takes in an image of a human face and

inputs it into a VGG19-based classifier that produces two outputs: sentiment and emotion

(Schibsted, 2024). The sentiment result consists of three metrics, the scores of which all sum to

1: “neutral”, “positive”, and “negative”. “Neutral” indicates neither warmth nor hostility. Rather,

it is the classifier’s indication that such facial expression does not convey a clear preference for

either the cooperative or competitive game strategy. On the other hand, the “positive” metric is

high for facial expressions of love, care, and affection, which our model interprets as markers of

cooperation. Finally, “negative” is associated with expressions such as sadness, disgust, and fear,

which our model perceives as indicators of competitiveness. While these three metrics can

sufficiently describe sentiment, the emotion output is composed of seven total parameters, the

sum of which also equals 1: “fear”, “sadness”, “surprise”, “neutral”, “happiness”, “anger”, and

“disgust”. It signals how much each of these emotions is conveyed by the facial expression on a

scale ranging from 0 to 1.

Now that we have metrics for sentiments and emotions associated with facial

expressions, we proceed to the Bayesian inference stage, which follows a similar algorithm as

the one used to classify animals as birds, fishes, reptiles, or mammals.

Initially, we assign a prior probability of 0.5 to both stealing and splitting, since either outcome is

equally likely a priori. Next, we transform the emotional scores assigned by the facial expression

classifier into true or false values that can be used by our Bayesian inference algorithm. To do so,

we adopted a thresholding strategy whereby if the model’s score for a certain emotion is greater

than a set value, we set such emotion to “true”. Otherwise, we leave it as “false”. The thresholds



<a name="br3"></a> 

adopted were the following: 0.2 for “fear”, 0.4 for “sadness”, 0.5 for “surprise”, 0.35 for

“happiness”, 0.15 for “anger”, and 0.2 for “disgust”. The logic behind assigning lower thresholds

to emotions such as fear, anger, and disgust is that conveying them even slightly already

indicates a strong willingness to engage in competitive behavior. This is especially true when

considering the interaction between emotion and sentiment, which will be explained shortly.

After thresholding, the model proceeds to the game theory-based decision making step,

which involves running Bayesian inference on the emotional categories that have been assigned

boolean values. In this step, emotions such as sadness and fear get assigned a higher “steal”

likelihood (0.65 and 0.8, respectively), while “happiness” receives a very high “split” probability

(0.9) and “surprise” receives a low probability for both outcomes (0.05). The logic behind setting

high “steal” probabilities for sadness and fear is that they are more closely associated with

competitive behavior and risk-taking, which we believe result in a greater willingness to steal.

This is corroborated by Johnson and Tversky (1983), who empirically demonstrate that

participants experience an increase in risk perception when experiencing sadness and fear.

Similarly, Harmon-Jones and Sigelman (2001) contend that physical displays of anger are

statistically significant indicators of a willingness to conquer, harm, or dominate other

individuals, explaining why we set a high “steal” likelihood (0.7) for anger in our Bayesian

inference framework. We also extrapolated this reasoning to the “disgust” category, to which we

assigned a likelihood of 0.6 for “steal”, since disgust demonstrates a condescending lack of

respect that is commonly associated with a proclivity to engage in selfish behavior.

In contrast, Rand et al. (2015) show that happiness elicits cooperation, which makes us

believe that someone displaying the happiness emotion would opt for splitting, since they would

operate under the assumption that the other game participant has a cooperative attitude that



<a name="br4"></a> 

favors risk aversion. For this reason, we have assigned a high “split” likelihood to the

“happiness” category (0.9). However, one can also express happiness by believing that they will

outsmart their opponent, which explains why we also set a high happiness value in the steal

category (0.4). Using these likelihood scores, the model creates a “posterior multiplier” variable

that is equal to the emotion-specific probability associated with each of the two possible

decisions (“split” or “steal”), or one minus this probability in case the boolean value for such

emotion is set to “false”. Then, we scale the “posterior multiplier” variable according to the

emotion being analyzed in the Bayesian inference step as follows:

● If the emotion is “happiness”, multiply the posterior multiplier by one plus the

difference between the positive and negative sentiment scores produced by the

facial expression detector. This is done to magnify the effects of a cooperative

attitude on the final result.

● If the emotion is “sadness”, multiply the posterior multiplier by one plus the

difference between the negative and positive sentiment scores produced by the

facial expression detector. This is the opposite strategy of the happiness

multiplier, and it is taken to magnify the effects of a competitive attitude on the

final result in case a sad facial expression is detected.

● If the emotion is “anger”, instead of multiplying by one plus the difference

between the negative and positive sentiment scores like in the “sadness” case,

multiply by two raised to one plus the difference between the negative and

positive sentiment scores. This is done to magnify the effect of competitiveness

on the final outcome even more, since anger is more conducive to risk-taking than

sadness (Hareli et al., 2021).



<a name="br5"></a> 

● If the emotion is “disgust”, we use almost the same logic as for “anger” but

instead of multiplying by two raised to one plus the difference between the

negative and positive sentiment scores, we raise 1.2 to one plus the difference

between the negative and positive sentiment scores. This is done to indicate a

smaller propensity towards competitiveness than for the “anger” emotion.

● If the emotion is “fear”, we use the same logic as “disgust” but instead of using a

base of 1.2 for the exponent, we use 3. This is done so that “fear” induces more

competitive behavior than either disgust or anger, as suggested by Hareli et al.

(2021).

● Lastly, for the “surprise” emotion, we multiply the posterior multiplier by Euler’s

number raised to the “neutral” sentiment score. This is done to provoke a mild

effect on the outcome, since although we use a relatively large multiplier (Euler’s

number), we raise it to the “neutral” sentiment, which received a generally lower

score than the other two (“positive” and “negative’) during the tests that we ran.

Furthermore, the reason behind using the neutral sentiment in this case is that

“surprise” is neither inherently positive or negative, but it is certainly not an

indicator of neutrality.

Equipped with this scaling strategy, we can now describe the entire Bayesian

inference-based algorithm. We start with prior probabilities of 0.5 for both stealing and splitting,

and set the initial value of “posterior” to such probabilities in both the “split” and “steal”

condition. Next, we iterate over the likelihood assigned to each emotion within the “split” and

“steal” categories. We set the value of “posterior multiplier” to such likelihood if such emotion is



<a name="br6"></a> 

present or to one minus such likelihood if the emotion is absent. Then, we scale the value of

“posterior multiplier” according to the logic described above and update the value of “posterior”

for either “split” or “steal” by computing its product with the scaled multiplier value. Once this

process has been repeated for both decision categories, a final posterior is obtained for each of

them, and the highest of the two indicates whether the model outputs “split” or “steal”.

Now that I have described how the model works, it is appropriate to discuss its strengths

and weaknesses. Clearly, our model’s strengths include its robustness to misleading facial

expression classification scores, which is accomplished through the emotional thresholding

strategy and the consideration of not just emotion but also sentiment during the Bayesian

inference step. Additionally, our model’s results reflect an extensive academic body of work that

perceives positive emotions such as happiness as indicators of cooperativeness and negative

emotions such as anger and fear as indicators of competitiveness. Our model accomplishes this

by not relying on the assumption that both players in “Split or Steal” would always behave

cooperatively, as would be the case for an approach based exclusively on game theory. Rather, it

considers facial expression as an important indicator of a participant’s decision.

Nonetheless, our model is not immune to weaknesses. Its biggest limitation is its

exclusive reliance on facial expressions, which are not perfect indicators of intent. Indeed, it is

possible for individuals to smile or laugh in ironic or condescending ways, which would not be

captured by the model, since these expressions —devoid of additional context— are commonly

associated with happiness. This example illustrates how the same facial expression could be

interpreted in ways that predict opposite game decisions (“steal” for irony and “split” for

happiness). In addition, our model’s lack of consideration for temporal data is also a limitation.

Due to the static nature of an image of a facial expression, it does not capture mood variations



<a name="br7"></a> 

during the game or additional data points that can signal intent. We believe that one way to

overcome this limitation is to create a model that also considers video data, but such

modification would be technically unfeasible within the short time frame assigned to this project.


**References**

Burton-Chellew, Maxwell N., et al. (2010). Cooperation in Humans: Competition between

Groups and Proximate Emotions. *Evolution and Human Behavior,* vol. 31, no. 2, 104–08.

<https://doi.org/10.1016/j.evolhumbehav.2009.07.005>

Hareli, S., Elkabetz, S., Hanoch, Y., and Hess, U. (2021). Social Perception of Risk-Taking

Willingness as a Function of Expressions of Emotions. *Frontiers in psychology*, 12, 655314.

<https://doi.org/10.3389/fpsyg.2021.655314>

Harmon-Jones, E., and Sigelman, J. (2001). State anger and prefrontal brain activity: Evidence

that insult-related relative left-prefrontal activation is associated with experienced anger and

aggression*. Journal of Personality and Social Psychology*, 80(5), 797–803.

<https://doi.org/10.1037/0022-3514.80.5.797>

Heffner, Joseph, and Oriel FeldmanHall. (2022). A Probabilistic Map of Emotional Experiences

during Competitive Social Interactions. *Nature Communications*, vol. 13, 1718.

<https://doi.org/10.1038/s41467-022-29372-8>

Hoegen, Rens, et al. (2018). The Impact of Agent Facial Mimicry on Social Behavior in a

Prisoner’s Dilemma. *Proceedings of the 18th International Conference on Intelligent Virtual*

*Agents,* ACM, 275–80. <https://doi.org/10.1145/3267851.3267911>



<a name="br11"></a> 

Johnson, E. J., and Tversky, A. (1983). Affect, generalization, and the perception of risk. *Journal*

*of Personality and Social Psychology,* 45(1), 20–31. <https://doi.org/10.1037/0022-3514.45.1.20>

Lerner, Jennifer S., and Larissa Z. Tiedens. (2006). Portrait of the Angry Decision Maker: How

Appraisal Tendencies Shape Anger’s Influence on Cognition. *Journal of Behavioral Decision*

*Making,* vol. 19, no. 2, 115–37. <https://doi.org/10.1002/bdm.515>

Mérő, L. (1998). John von Neumann’s Game Theory. In: Moral Calculations. *Springer,* New

York, NY. <https://doi.org/10.1007/978-1-4612-1654-4_6>

Parks, Craig D., and Lorne G. Hulbert. (1995). High And Low Trusters’ Responses To Fear in a

Payoff Matrix. *Journal of Conflict Resolution,* vol. 39, no. 4, 718–30.

<https://doi.org/10.1177/0022002795039004006>

Raghunathan, R., and Pham, M. T. (1999). All negative moods are not equal: Motivational

influences of anxiety and sadness on decision making. *Organizational Behavior and Human*

*Decision Processes*, 79(1), 56–77. <https://doi.org/10.1006/obhd.1999.2838>

Rand, David G., et al. (2015). The Collective Benefits of Feeling Good and Letting Go: Positive

Emotion and (Dis)Inhibition Interact to Predict Cooperative Behavior. *PLoS ONE*, vol. 10, no. 1,

0117426\. <https://doi.org/10.1371/journal.pone.0117426>



<a name="br12"></a> 

Schibsted. (2024). Facial Expression Classifier. *Hugging Face.*

<https://huggingface.co/spaces/schibsted/facial_expression_classifier>

Shen, Xunbing, et al. (2021). Catching a Liar Through Facial Expression of Fear. *Frontiers in*

*Psychology,* vol. 12, 675097. <https://doi.org/10.3389/fpsyg.2021.675097>

Van Den Assem, Martijn J., et al. (2012). Split or Steal? Cooperative Behavior When the Stakes

Are Large. *Management Science,* vol. 58, no. 1, 2–20. <https://doi.org/10.1287/mnsc.1110.1413>



<a name="br13"></a> 

**Appendix**

AI Tool Usage and Non-Usage Disclosure: I did not utilize AI tools while writing this final

report, since I did not find them necessary due to my extensive knowledge of the methods and

academic literature underlying my group’s cognitive model.

Link to project code:

<https://github.com/lrmantovani10/Split_or_Steal_Cog_Model/blob/main/app.py>

