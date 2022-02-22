# Python JPEG decoder / talk sources

## The decoder

The decoder is written in Python, tested against Python 3.10. It uses some newer features from the `typing` module so older versions probably won't work.

The only external dependencies required are `numpy` and `Pillow`.

### Running the decoder

`python3.10 -i decode.py subject.jpg` will drop you into a shell where you can play with the PIL image and the intermediate metadata structures. Try `image_segments` to get a list of all segments, or `image_metadata` to get the consolidated, parsed metadata structure. You can also look at `decode_animation.py` as an example of what kind of things you can do using the parsed results.

`python3.10 decode.py subject.jpg output.png` will write the PIL image to a PNG file.

Note: random JPG files you find on the internet probably won't work with this decoder. The feature set is very small, and it still has a lot of bugs. It's talk-ware, for sure :)


## Output generation

```
$ npx @marp-team/marp-cli@latest slides.md
```