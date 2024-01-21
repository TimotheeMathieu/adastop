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
             (commit "f609b94924aad8ccace108f3daba8e856fb306e9")))
       (file-name (git-file-name name version))
       (sha256
               (base32
                "1idxz6d3lg6l6ib75dcypnq6nb4frfqxy82lckfnfxz23npjjkwv"))
       )
     )
    (build-system python-build-system)
    (propagated-inputs (list python-click python-matplotlib python-numpy python-pandas python-tabulate))
    (home-page "https://github.com/TimotheeMathieu/adastop")
    (synopsis "Sequential testing for efficient and reliable comparison of stochastic algorithms. ")
    (description
     "Sequential testing for efficient and reliable comparison of stochastic algorithms.")
    (license license:expat)))


