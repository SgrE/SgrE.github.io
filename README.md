www.lfhsgre.org

This homepage uses a new design based on [jemdoc](https://jemdoc.jaboc.net/), following 

**Demo:**  [https://szl2.github.io/jemdoc-new-design/www/index.html](https://szl2.github.io/jemdoc-new-design/www/index.html)

## Usage

Assuming you have already install the  `jemdoc+MathJax`, notice that

in `./jemdoc_files/`, we store `.jemdoc` files and `mysite.conf`.

Suppose you are currently in `./jemdoc_files/`, we use the following to compile

```
python3 ../jemdoc -c mysite.conf -o ../  *.jemdoc
```

You can also use this for single page generation or all page generation by using `*.jemdoc`.


