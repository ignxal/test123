from textblob import TextBlob
import twily_classifier as cl
import stop_words as stopwords
import json

with open('twilybot.json', 'r') as f:
    array = json.load(f)

CONVERSATION = array["conversations"]
BOT_NAME = 'Twily'
STOP_WORDS = stopwords.sw_list
#holds the negative sentiment floating point value. monitor user sentiment index
neg_distribution = []

#Appends the neg_distribution list with negative probability and returns the appended value.
def sentiment(u_input):
    """Auxiliary function: Appends 'neg_distribution'
    with negative probability also returns Negative Probability"""

    blob_it = cl.trainer().prob_classify(u_input)
    npd = round(blob_it.prob("neg"), 2) #extract neg values, roundedup
    neg_distribution.append(npd) #upd list
    return npd

#Implements a rule-based bot. It takes the user input in the form of a string and in sequence it pre-processes the input string, converts it to lowercase, tokenizes it and removes stop words.
#It then iterates through CONVERSATION, if filtered_input intersects response_set is updated.
#Returns: If the set is empty a message, else it returns the longest string in the set
def simplebot(user_input):
    """Rule base bot, takes an argument, user input in form of a string. (truncated)"""

    user_blob = TextBlob(user_input)
    lower_input = user_blob.lower()
    token_input = lower_input.words
    #list of words not listed in STOP_WORDS from textblob object
    filtered_input = [w for w in token_input if w not in STOP_WORDS]
    #empty set to be updated with all the possible matches returned by our set intersection of user input and CONVERSATION.
    response_set = set()
    for con_list in CONVERSATION:
        for sentence in con_list:
            sentence_split = sentence.split()
            if set(filtered_input).intersection(sentence_split):
                response_set.update(con_list)
    if not response_set:
        return "I am sorry, I don't have an answer, ask again"
    else: 
        return max(response_set, key=len)

#This function takes an argument user_input in the form of a string and calls sentiment() to monitor the user sentiment index.
#If the emotional index, set by sentiment() and taken from neg_distribution, increases above a set threshold and is sustained, an automatic response/action is triggered.
#The function also sends user_input to simplebot() to generate a chatbot response.

def escalation(user_input):
    """ Takes an argument, user_input, in form of a string ..."""

    live_rep = f"We apologize {BOT_NAME} is unable to assist you, we are getting a live representative for you, please stay with us ..."

    sentiment(user_input)
    list_len = len(neg_distribution)
    bot_response = simplebot(user_input)
    if list_len > 3:
        last_3 = neg_distribution[-3:]
        if last_3[0] > .40 and last_3[0] <= last_3[1] <= last_3[2]:
            return live_rep
        else:
            return bot_response
    else:
        return bot_response
#test
if __name__ == '__main__':
    while True:
        try:
            user_input = input('You: ')
            print(escalation(user_input))
            print(neg_distribution)
        except (KeyboardInterrupt, EOFError, SystemExit):
            break