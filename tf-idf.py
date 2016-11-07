import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer


def calculate_tfidf(args):
    data_dir = args.directory
    # Get a list of the file names in the directory.
    names = [file[:-4] for file in os.listdir(data_dir) if file[-4:] == '.txt']
    # Get a list of files to feed to scikit-learn.
    files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file[-4:] == '.txt']
    # Chomp chomp -- getting trigrams
    tf = TfidfVectorizer(input='filename', analyzer='word', ngram_range=(3, 3), min_df=0, smooth_idf=False, sublinear_tf=True)
    tfidf_matrix = tf.fit_transform(files)
    # These are the actual phrases
    feature_names = tf.get_feature_names()
    # These are the scores
    texts = tfidf_matrix.todense()
    for index, row in enumerate(texts):
        name = names[index]
        print '\n\n{}\n'.format(name.upper())
        text = row.tolist()[0]
        # If the score is not 0 save it with an index (which will let us get the feature_name)
        scores = [pair for pair in zip(range(0, len(text)), text) if pair[1] > 0]
        sorted_scores = sorted(scores, key=lambda t: t[1] * -1)
        # Print the top 20 results for each file
        for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_scores][:20]:
            print('{0: <40} {1}'.format(phrase.encode('utf-8'), score))


def main():
    parser = argparse.ArgumentParser(description="Calculate TF-IDF values.")
    parser.add_argument('directory', help='Full path of a directory containing text files.')
    args = parser.parse_args()
    calculate_tfidf(args)


if __name__ == "__main__":
    main()
