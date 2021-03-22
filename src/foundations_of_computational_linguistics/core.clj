(ns foundations-of-computational-linguistics.core)

;; * `strings` is like a sentence
;; * To characterize infinete languages, we can use 2 approaches: recognition and generation
;;   * recognizer: takes a string and returns a the string is in the language or not
;;   * generator: takes a number and returns a string which is in the language
;; * generator is nondeterministic
;;   * it takes an arugument `n` and generate a n-length string
;;   * and we don't know how the `n` was selected
;; * `flip` function defines a probability distribution
;;   * `flip` can be used as probability distribution to generator
;; * sampler uses generator and a probaility distribution to draw samples from a language, say (ab)*
;;   * sampler is _probabilistic generator_ (a probalistic version of the generator)
;; * we can also write the probabilistc equivalent to recognizer for a language: _scorer_
;; * recognizer and generator's probabilistic versions are scorer and sampler
;; * p (s) ∈ [0,1] and ∑s∈Sp (s) =1
;;   * S is discrete probability distibutions as a set which is called the support, sample space or set ouf outcomes

;; memo
;; * a formal language is a _model_?
;;   * -> see 5. Formal Languages
;;   * we want to build up mathematically precise models to explain gramaticality

(defn generate-ab* [n]
  (if (= n 0)
    '()
    (concat '(a b) (generate-ab*  (dec n)))))

(defn flip
  ([] (flip 0.5))
  ([weight] (< (rand 1) weight)))

(defn sample-n []
  (if (flip)
    0
    (inc (sample-n))))

(defn sample-ab*  [] (generate-ab*  (sample-n)))

(defn sample-corpus [generator size]
  (if (= size 0)
    '()
    (cons (generator)
          (sample-corpus generator (- size 1)))))

#_(sample-corpus sample-ab* 4)

#_(defn score-corpus [scorer corpus]
    (if (empty? corpus)
      1.0
      (* (scorer (first corpus))
         (score-corpus scorer (rest corpus)))))

(defn prefix? [pr str]
  (cond
    (> (count pr) (count str)) false
    (empty? pr) true
    (= (first pr) (first str)) (prefix? (rest pr) (rest str))
    :else false))

(defn score-ab* [str]
  (if (empty? str)
    0.5
    (if (prefix? '(a b) str)
      (* 0.5 (score-ab* (rest (rest str))))
      0)))

(def my-corpus '((a b)
                 ()
                 (a b a b)))

#_(score-corpus score-ab* (sample-corpus sample-ab* 4))

;; compare this to `sample-ab*`
(defn sample-kleene-ab []
  (if (flip)
    '()
    (cons (if (flip) 'a 'b)
          (sample-kleene-ab))))

#_(sample-kleene-ab)

;; 0.5 came from `flip`'s _p_
(defn score-kleene-ab [s]
  (if (empty? s)
    0.5
    (* 0.5 (if (= (first s) 'a)
             0.5
             0.5)
       (score-kleene-ab (rest s)))))

#_(list [:empty (score-kleene-ab '())]
        [:ab (score-kleene-ab '(a b))]
        [:ababab (score-kleene-ab '(a b a b a b))])

;; {a,b}* assigns probability mass to a much larger _set_ of strings
;; than the probabilistic formal language defined on (ab)*
#_(list
   [:epsilon (score-ab* '()) (score-kleene-ab '())]
   [:ab (score-ab* '(a b)) (score-kleene-ab '(a b))]
   [:ababab (score-ab* '(a b a b a b))  (score-kleene-ab '(a b a b a b))]
   [:bb (score-ab* '(b b)) (score-kleene-ab '(b b))]
   [:abb (score-ab* '(a b b)) (score-kleene-ab '(a b b))])

;; (flip p) is a random varible and (list (flip p) (flip p)) is also a random variable
;; the latter is a complex random variable
#_(def mu-random-variable
    (let [p 0.5]
      (list (flip p) (flip p))))

;; categorical distribution
(defn normalize [params]
  (let [sum (apply + params)]
    (map (fn [x] (/ x sum)) params)))

#_(normalize 2 1 1)

(defn sample-categorical [outcomes params]
  (if (flip (first params))
    (first outcomes)
    (sample-categorical (rest outcomes)
                        (normalize (rest params)))))

(defn score-categorical [outcome outcomes params]
  (cond
    (empty? params) (throw "no matching outcome")
    (= outcome (first outcomes)) (first params)
    :else (score-categorical outcome
                             (rest outcomes)
                             (rest params))))

#_(score-categorical 'me
                     '(call me Ishmael)
                     '(1/3 1/3 1/3))

(defn log2 [n]
  (/ (Math/log n)
     (Math/log 2)))

#_(defn logsumexp [& log-vals]
    (log2
     (apply +
            (map (fn [z]
                   (Math/pow 2 z))
                 log-vals))))

;; avoiding underflow 🤔
(defn logsumexp [& log-vals]
  (let [mx (apply max log-vals)]
    (+ mx
       (log2
        (apply +
               (map (fn [z] (Math/pow 2 z))
                    (map (fn [x] (- x mx))
                         log-vals)))))))

(defn uniform-distribution [vocab]
  (let [n (count vocab)
        p (/ 1 n)]
    (take n (repeat p))))

;; bag of words (BOW) approach
;; we assume each words is generated independently from some distribution without reference to the other words.
(defn sample-BOW-sentence [vocab len]
  ;; is it better the probability of a vocablary is calculated from it (not as an argument)?
  ;; because BOW is defined as such
  (let [prob (uniform-distribution vocab)]
    (if (zero? len)
      '()
      (cons (sample-categorical vocab prob)
            (sample-BOW-sentence vocab (dec len))))))

;; * unfold: a generator for structured objects like lists
(defn list-unfold [generator len]
  (if (zero? len)
    '()
    (cons (generator)
          (list-unfold generator (dec len)))))

;; if you take a `uniform-distribution` as an argument, the function is more generalized?
#_(defn sample-BOW-sentence [vocab len]
    (list-unfold #(sample-categorical vocab
                                      (uniform-distribution vocab))
                 len))

;; * fold is a inverse operation of unfold
(defn list-foldr [f base lst]
  (if (empty? lst)
    base
    (f (first lst)
       (list-foldr f base (rest lst)))))

(defn score-BOW-sentence [sen vocablary]
  (let [probabilities (uniform-distribution vocablary)]
    (list-foldr
     (fn [word rest-score]
       (* (score-categorical word vocablary probabilities)
          rest-score))
     1
     sen)))

#_(score-BOW-sentence '(call me Ishmael) '(call me Ishmael))

(defn score-corpus [corpus vocablary]
  (list-foldr
   (fn [sen rst]
     (* (score-BOW-sentence sen vocablary) rst))
   1
   corpus))

;; A corpus C is a multi-set of strings (e.g. real world data)
#_(score-corpus
   '((Call me Ishmael)
     (Some years ago - never mind how long precisely
           - having little or no money in my purse |,|
           and nothing particular to interest me on shore
           |,| I thought I would sail about a little and
           see the watery part of the world))
   '(Call me Ishmael Some years ago - never
          mind how long precisely -
          having little or no money in my purse |,|
          and nothing particular to interest on shore
          I thought would sail about a little see
          the watery part of the world))
