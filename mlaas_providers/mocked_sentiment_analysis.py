from mlaas_providers.sentiment_analysis import SentimentAnalysis
import queue
from utils.requestQueue import RequestQueue
import random

class MockedSentimentAnalysis(SentimentAnalysis):
    def arrange_results(self, result_list):
        result_list.sort(key=lambda x: x["index"], reverse=False)

        data = list(map(lambda x: x["predictions"], result_list))

        return data

    def call_mocked_sentiment(self, review, client, result_queue):
        try:
            result = client(review["document"])
            result_data = {'sentiment': result}

            result_queue.put({"predictions":result_data, "index":review["index"]})
        except Exception as e:
            print("Error:", e)
            print(review["document"])
            raise e

    def run_naive_classifier(self, review):
        classes=['negative', 'neutral', 'positive']
        # use the code above to simulate network latence
        # sleep(2)
        # use the code above to simulate api errors
        # if(random.randint(1, 20) == 1):
        #     raise Exception("An error occurred (simulation)")
        
        result_index = random.randint(-1, 1) 
        return classes[result_index]

    def classify(self, documents):
        
        documents = [{"document":doc, "index": i} for i,doc in enumerate(documents)] # dict

        result_queque = queue.Queue()
        request_queue = RequestQueue(function_to_call=self.call_mocked_sentiment,
                                    iterate_args=documents,
                                    fixed_args=[self.run_naive_classifier, result_queque],
                                    call_rate=6000)
        request_queue.run()

        results = []
        for i in range(len(documents)):
            results.append(result_queque.get())
        
        results = self.arrange_results(results)
        results = list(map(lambda r: r['sentiment'], results))
        
        return results
