
class Tester:

    def __init__(self, k=[5, 10, 20]):
        self.k = k
        self.initialize()

    def initialize(self):
        self.recall = [0]*len(self.k)
        self.mrr = [0]*len(self.k)
        self.evaluation_count = 0

    def get_rank(self, target, predictions):
        for i in range(len(predictions)):
            if target == predictions[i]:
                return i+1

        raise Exception("could not find target in sequence")

    def update_higher_k(self, j, inv_rank):
        for i in range(j+1, len(self.k)):
            self.recall[i] += 1
            self.mrr[i] += inv_rank

    def evaluate_sequence(self, predicted_sequence, target_sequence, seq_len):
        for i in range(seq_len):
            target_item = target_sequence[i]
            k_predictions = predicted_sequence[i]

            for j in range(len(self.k)):
                k = self.k[j]
                if target_item in k_predictions[:k]:
                    self.recall[j] += 1
                    inv_rank = 1.0/self.get_rank(target_item, k_predictions[:k])
                    self.mrr[j] += inv_rank

                    self.update_higher_k(j, inv_rank)
                    break

            self.evaluation_count += 1


    def evaluate_batch(self, predictions, targets, sequence_lengths):
        for batch_index in range(len(predictions)):
            predicted_sequence = predictions[batch_index]
            target_sequence = targets[batch_index]
            self.evaluate_sequence(predicted_sequence, target_sequence, sequence_lengths[batch_index])

    def get_stats(self):
        message = ""
        for i in range(len(self.k)):
            k = self.k[i]
            
            recall_k = self.recall[i]/self.evaluation_count
            mrr_k = self.mrr[i]/self.evaluation_count

            message += "\tRecall@"+str(k)+" =\t"+str(recall_k)+"\n"
            message += "\tMRR@"+str(k)+" =\t"+str(mrr_k)+"\n"
        
        recall5 = self.recall[0]/self.evaluation_count
        recall20 = self.recall[2]/self.evaluation_count

        return message, recall5, recall20


    def get_stats_and_reset(self):
        message = self.get_stats()
        self.initialize()
        return message
