class SentimentAnalysis:
    def chunks(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def ensure_limits(self, batch):
        safe_batch = []
        for comment in batch:
            if isinstance(comment, str) and len(comment) > 0:
                if len(comment.encode('utf-8')) < 5000:
                    safe_batch.append(comment)
                else:
                    safe_batch.append(comment[:(self.MAX_COMMENT_SIZE - 1)])
        return safe_batch

