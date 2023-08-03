import pickle
import numpy as np

with open('bagOfWords.pickle', 'rb') as f:
    lr = pickle.load(f)
    mapping = pickle.load(f)

def preprocess_text(headline, body):
    # Combine headline and body into a single string
    text = f"{headline} {body}"
    # Gets the mapping dictionary from data
    count_vector = np.zeros(len(mapping))
    #makes the bag of word from that mapping dictionary
    for word in text.split():
        if word in mapping:
            count_vector[mapping[word]] += 1
    #reshapes data
    new_text_vector=count_vector.reshape(1, -1)
    # Make a prediction using the trained classifier
    predicted_label = lr.predict(new_text_vector)
    # Make predictions of confidence
    predicted_confidence = lr.predict_proba(new_text_vector)

    # Get the confidence 
    confidence = round(predicted_confidence.max()*100 ,2 )

    #return both
    return [predicted_label[0],confidence]

# Example new text data
new_headline = "Pastor Paul Mackenzie: What did the starvation cult leader preach?"
new_body = '''
The leader of a Christian cult in Kenya is to remain in police custody for another month, as the exhumation of bodies found in mass graves on his land continues. At least 130 have been discovered so far.

Pastor Paul Nthenge Mackenzie has said he closed down his Good News International Church four years ago after nearly two decades of operation.

But the BBC has uncovered hundreds of his sermons still available online, some of which appear to have been recorded after this date.

What picture do they paint of a man whose followers have starved themselves to death?

'Let no-one turn back'
In a passionate, raspy voice, Pastor Mackenzie delivers his sermons to large congregations in thrall to his apocalyptic themes.

"We are about to win the battle… let no-one turn back… the journey is about to be accomplished," reads a banner across the screen.

One series of videos on his church's YouTube channel has the caption: "End Time Kids" and shows groups of young children delivering messages to the camera.

Screenshot of Pastor Mackenzie delivering a sermon
IMAGE SOURCE,YOUTUBE
Image caption,
Apocalyptic themes and warnings about impending doom feature heavily in Pastor Mackenzie's sermons
Others culminate in exorcisms in which followers - often women - writhe around on the ground while he "torments" the demonic forces within them.

These YouTube channels have thousands of subscribers and a Facebook page set up by his church links to many of the videos.

The channels, which the BBC flagged to YouTube after Pastor Mackenzie's arrest, remain active. The BBC has observed several monetised videos on the channels, meaning YouTube makes money from the videos via online ads.

Google and Meta have not responded to a BBC request for comment.

It's not clear when the sermons were filmed, but there is reference to an upcoming preaching event by Pastor Mackenzie in Nairobi in January 2020, which contradicts his claim to have ended his preaching activities the previous year.

'Children are crying because they are hungry, let them die'
Former members of the church have claimed that they were forced to fast as part of their adherence to its teachings.

There is no direct evidence in the dozens videos we've seen of Pastor Mackenzie directly ordering people to fast, but there are many references to followers sacrificing what they hold dear, including their lives.

Kenyans seek relatives among starvation cult victims in Kilifi
IMAGE SOURCE,REUTERS
Image caption,
Grieving relatives of the cult victims in the Kenyan coastal town of Kilifi
"There are people who don't even want to preach [about] Jesus. They say their children are crying because they are hungry, let them die. Is there a problem there?"

In an interview with the Kenyan Nation newspaper a few weeks ago, Pastor Mackenzie denied he forced his followers to fast.

"Is there a house maybe or an enclosure or a fence somewhere that has been found [at the farm] where people might have been locked in?" he replied when the reporter asked him about this.

'Education is evil'
Another theme of Pastor Mackenzie's sermons has been the idea that formal education is satanic and used to extort money.

"They know education is evil. But they use it for their own gains" he says in one sermon. "Those who sell uniforms, write books…those who make pens… all kinds of rubbish. They use your money to enrich themselves while you become poor."

In 2017 and again in 2018, he was arrested for encouraging children not to go to school as he claimed education was "not recognised in the Bible".

Pastor Mackenzie has also condemned education for promoting homosexuality through sex education programmes.

"I told people education is evil…. Children are taught gayism and lesbianism,'' he told the Nation newspaper.

Doctors 'serve a different God'
He has also encouraged mothers to avoid seeking medical attention during childbirth and not to vaccinate their children.

In one of the videos, a woman narrates how she helped to deliver a baby through prayer and without the need for a caesarean section, adding that she later received a "prompting" from the holy spirit to warn her neighbour against vaccinating her child.

The pastor then echoes her sentiments that vaccines are not necessary, claiming that doctors "serve a different God".

He also discourages women from plaiting their hair, wearing wigs and wearing ornaments.

Satanic symbols and global conspiracies
Much of Pastor Mackenzie's preaching relates to the fulfilment of Biblical prophecies about Judgement Day.

The church's online content also features posts about the end of the world, impending doom and the supposed dangers of science.

And there are frequent warnings of an omnipotent satanic force that has supposedly infiltrated the highest echelons of power around the world.

Screenshot of conspiratorial memes on the church's Facebook page
IMAGE SOURCE,FACEBOOK
Image caption,
The church's online content features a wide range of conspiratorial images and memes
He repeatedly references "New World Order" - a conspiracy theory about a plot by global elites to bring about an authoritarian world government, replacing nation states - falsely claiming the Catholic Church, the UN and the US are behind it.

He is also highly sceptical of modern technology, previously claiming a plan by the Kenyan government to establish a unique identity number for citizens to access government services was the 'mark of the beast'.
'''


print(preprocess_text(new_headline, new_body))


#returned fake