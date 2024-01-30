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
             (commit "8d9585eb0387ced1225dce20929337cb59a2e247")))
       (file-name (git-file-name name version))
       (sha256
               (base32
                "1jc0441fkkifm0f2vhnfr2l4r2yykqswyqp04h475kb55ii5n7nv"))
       )
     )
    (build-system python-build-system)
    (propagated-inputs (list python-click python-matplotlib python-numpy python-pandas python-tabulate))
    (home-page "https://github.com/TimotheeMathieu/adastop")
    (synopsis "Sequential testing for efficient and reliable comparison of stochastic algorithms. ")
    (description
     "Sequential testing for efficient and reliable comparison of stochastic algorithms.")
    (license license:expat)))


