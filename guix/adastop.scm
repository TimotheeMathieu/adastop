(define-module (adastop)
  #:use-module ((guix licenses) #:prefix license:)
  #:use-module (guix git-download) 
  #:use-module (guix packages)
  #:use-module (gnu packages web)
  #:use-module (gnu packages python-web)
  #:use-module (gnu packages python-science)
  #:use-module (guix build-system python)
  #:use-module (gnu packages check)
  #:use-module (gnu packages python-xyz))

(define-public python-adastop
  (package
    (name "python-adastop")
    (version "0.1.1")
    (source
     (origin
       (method git-fetch)
       (uri (git-reference
             (url "https://github.com/TimotheeMathieu/adastop")
             (commit "142650e56fcb797e29661433d8e936f3aa221af4")))
       (file-name (git-file-name name version))
       (sha256
               (base32
                "0jrfm36r9p398mnr6d77srhms2jrl8i6rs69r4pb1v91b0ig43jb"))
       )
     )
    (build-system python-build-system)
    (propagated-inputs (list python-click python-matplotlib python-numpy python-pandas python-tabulate))
    (home-page "https://github.com/TimotheeMathieu/adastop")
    (synopsis "Sequential testing for efficient and reliable comparison of stochastic algorithms. ")
    (description
     "Sequential testing for efficient and reliable comparison of stochastic algorithms.")
    (license license:expat)))


