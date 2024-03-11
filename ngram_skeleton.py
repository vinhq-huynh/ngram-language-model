import math, random

################################################################################
# Part 0: Utility Functions
################################################################################
COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']


def start_pad(n):
    """ Returns a padding string of length n to append to the front of text
    as a pre-processing step to building n-grams """
    return '~' * n


def ngrams(n, text):
    """ Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character """
    # Initialize an empty list to store the ngrams
    ngrams = []
    text = start_pad(n) + text

    # Loop through the text starting from the nth character
    for i in range(n, len(text)):
        # Find the context, which is the n characters before the i (current) character
        context = text[i - n:i]
        # The current character
        char = text[i]
        # Append the ngram to the list
        ngrams.append((context, char))

    return ngrams


# Example usage
text = "hello world"
n = 3
print(ngrams(n, text))


def create_ngram_model(model_class, path, n=2, k=0):
    """ Creates and returns a new n-gram model trained on the city names
    found in the path file """
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model


def create_ngram_model_lines(model_class, path, n=2, k=0):
    """ Creates and returns a new n-gram model trained on the city names
    found in the path file """
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model


################################################################################
# Part 1: Basic N-Gram Model
################################################################################
class NgramModel(object):
    """ A basic n-gram model using add-k smoothing """
    def __init__(self, n, k):
        self.n = n  # Context length
        self.k = k  # For add-k smoothing
        self.ngrams = {}  # Dictionary to store n-gram counts
        self.vocab = set()  # Set to store unique characters (the vocabulary)

    def get_vocab(self):
        """ Returns the set of characters in the vocab """
        return self.vocab

    def update(self, text):
        """ Updates the model n-grams based on text """
        # Update vocab
        self.vocab.update(set(text))
        # Pad the text with '~' characters
        text = start_pad(self.n) + text
        # Update ngram counts
        for i in range(self.n, len(text)):
            context = text[i - self.n:i]
            char = text[i]
            if context not in self.ngrams:
                self.ngrams[context] = {}
            if char not in self.ngrams[context]:
                self.ngrams[context][char] = 0
            self.ngrams[context][char] += 1

    def prob(self, context, char):
        """ Returns the probability of char appearing after context """
        # If the context is novel, return uniform probability
        if context not in self.ngrams:
            return 1 / len(self.vocab)
        # Calculate probability of char given context
        context_count = sum(self.ngrams[context].values())
        char_count = self.ngrams[context].get(char, 0) + self.k  # Applying add-k smoothing
        return char_count / (context_count + self.k * len(self.vocab))  # Also add-k for each char in the denominator

    def random_char(self, context):
        """ Returns a random character based on the given context and the
        n-grams learned by this model """
        # Get the sorted list of vocab characters
        v = sorted(list(self.vocab))
        # Generate a random number r
        r = random.random()
        # Cumulative probability
        cum_prob = 0.0
        # Find the character that satisfies the inequality in the prompt
        for char in v:
            cum_prob += self.prob(context, char)
            if cum_prob > r:
                return char

    def random_text(self, length):
        """ Returns text of the specified character length based on the
        n-grams learned by this model """
        if self.n == 0:
            context = ''  # Context is empty string if n is 0
        else:
            context = '~' * self.n  # Start with a context of n '~' characters

        result = ''  # Initialize the result string
        for i in range(length):
            char = self.random_char(context)
            result += char
            if self.n > 0:
                context = context[1:] + char  # Update the context
        return result

    def perplexity(self, text):
        """ Returns the perplexity of text based on the n-grams learned by
        this model """
        # Pad the text with '~' characters
        text = start_pad(self.n) + text
        # Calculate perplexity using logs to avoid underflow
        log_prob_sum = 0
        for i in range(self.n, len(text)):
            context = text[i - self.n:i]
            char = text[i]
            prob = self.prob(context, char)
            if prob > 0:
                log_prob_sum += math.log(prob)
            else:
                return float('inf')  # Return positive infinity if any zero probabilities are encountered
        # Normalize by the number of characters to get perplexity
        perplexity = math.exp(-log_prob_sum / len(text))
        return perplexity


################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.ngrams = []
        self.vocab = []
        self.w = 1 / (self.n + 1)

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        print(self.vocab)
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        for char in text:
            if char not in self.vocab:
                self.vocab.append(char)

        for m in range(0, self.n+1):
            new_ngrams = ngrams(m, text)
            try:
                self.ngrams[m] += new_ngrams
            except:
                self.ngrams.append([])
                self.ngrams[m] += new_ngrams
        
            
    def prob_helper(self, context, char, m):
        ''' Returns the probability of char appearing after context '''
        
        count = 0
        total = 0
        for ngram in self.ngrams[m]:
            if ngram[0] == context:
                total += 1
                if ngram[1] == char:
                    count += 1
        try:
            return (count + self.k) / (total + (self.k * len(self.vocab)))
        except:
            return 1 / len(self.vocab)
    
    def prob(self, context, char):
        accum = 0
        for m in range(self.n, -1, -1):
            accum += (self.w * self.prob_helper(context[(self.n-m):], char, m))
        return accum

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################
class ClassifyCity:
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.models = {country_code: NgramModel(self.n, self.k) for country_code in COUNTRY_CODES}

    def train(self, training_folder):
        # For each country, train a language model on the city names
        for country_code in COUNTRY_CODES:
            file_path = training_folder + '/' + country_code + '.txt'
            with open(file_path, 'r', encoding='ISO-8859-1') as file:  # Somehow utf-8 encoding gives me error reading the file
                city_names = file.read().splitlines()  # Split city names by line
                for city_name in city_names:
                    self.models[country_code].update(city_name + '\n')  # Adding end-of-text character

    def classify(self, city_name):
        # Classify a new city name
        likelihoods = {
            country_code: model.perplexity(city_name + '\n')  # Use perplexity to compare models
            for country_code, model in self.models.items()
        }
        return min(likelihoods, key=likelihoods.get)  # Return the country with the lowest perplexity

    def classify_test_file(self, test_file_path, output_file_path):
        with open(test_file_path, 'r', encoding='utf-8', errors='ignore') as file:
            city_names = file.read().splitlines()

        # Classify each city name
        predictions = [self.classify(city_name) for city_name in city_names]

        # Write predictions to the output file
        with open(output_file_path, 'w', encoding='utf-8') as out_file:
            for prediction in predictions:
                out_file.write(prediction + '\n')


if __name__ == '__main__':
    print("Part 1: Generating Shakespeare with no smoothing, ngram order 7:")
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7, 0)
    print(m.random_text(250))

    print("Part 2. Testing perplexity of some examples from the prompt")
    m = NgramModel(1, 0)
    m.update('abab')
    m.update('abcd')
    print(m.perplexity('abcd'))
    print(m.perplexity('abca'))
    print(m.perplexity('abcda'))

    print("Part 3. Testing perplexity of some examples from the prompt using interpolation method.")
    m = NgramModelWithInterpolation(1, 0)
    m.update('abab')
    m.update('abcd')
    print(m.perplexity('abcd'))
    print(m.perplexity('abca'))
    print(m.perplexity('abcda'))

    print("Part 3. input file: cities_test.txt. output file: test_labels.txt")
    #embed = discord.Embed()
    classifier = ClassifyCity(n=3, k=1)
    classifier.train('train')
    classifier.classify_test_file('cities_test.txt', 'test_labels.txt')


