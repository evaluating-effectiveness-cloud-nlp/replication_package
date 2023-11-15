from google.cloud import language_v1
from mlaas_providers.sentiment_analysis import SentimentAnalysis
import queue
from utils.requestQueue import RequestQueue

text_type = language_v1.Document.Type.PLAIN_TEXT

def map_sentiment(sentiment):
    if(sentiment.score > 0):
        return 'positive'
    elif sentiment.score < 0:
        return 'negative'
    else:
        return 'neutral'

def call_google_sentiment(review, client, result_queue):
    try:
        print('\r'+str(len(result_queue.queue)), end='')
        # docs: cloud.google.com/natural-language/docs/basics
        document = language_v1.Document(content=review["document"], type_=text_type, language="EN")
        result = client.analyze_sentiment(request={'document': document})
        result_data = map_sentiment(result.document_sentiment)
        result_data = {'sentiment': map_sentiment(result.document_sentiment)}

        result_queue.put({"predictions":result_data, "index":review["index"]})
        # result_queue.put(result_data)
    except Exception as e:
        print("Error:", e)
        print(review["document"])
        raise e

def arrange_results(result_list):
    result_list.sort(key=lambda x: x["index"], reverse=False)

    data = list(map(lambda x: x["predictions"], result_list))

    return data
class GoogleSentimentAnalysis(SentimentAnalysis):

    def __init__(self):
        self.client = language_v1.LanguageServiceClient()
        self.MAX_COMMENT_SIZE = 5000

    def classify(self, documents):
        
        documents = [{"document":doc, "index": i} for i,doc in enumerate(documents)]# dict

        result_queque = queue.Queue()
        request_queue = RequestQueue(function_to_call=call_google_sentiment,
                                    iterate_args=documents,
                                    fixed_args=[self.client, result_queque],
                                    call_rate=600)
        request_queue.run()

        print('\r', end='')
        
        results = []
        for i in range(len(documents)):
            results.append(result_queque.get())
        
        results = arrange_results(results)
        results = list(map(lambda r: r['sentiment'], results))
        
        return results
