from django.core.cache import cache
import pickle

gen_cache_key = 'generator_cache'
# this key is used to `set` and `get` your trained model from the cache

train_generator = cache.get(gen_cache_key)

if train_generator is None:
    # your model isn't in the cache
    # so `set` it
    # load the pickle file
    train_generator = pickle.load(open('polls/image-generator-model.pkl', 'rb'))
    cache.set(gen_cache_key, train_generator, None)
