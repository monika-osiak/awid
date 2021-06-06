W celu uruchomienia zestawu testów należy uruchomić REPL juli. Następnie niezbędne jest wywołanie komend
```julia-repl
] activate {{ scieżka do projektu}}
] instantiate
```
Pierwsza uruchamia środowisko dla projektu, druga pobiera wymagane zależności.
> Jeżeli nie została zdefiniowana zmienna środowiskowa "TEST_PERFORMANCE" uruchomione zostaną tylko testy sprawdzające poprawność
> Można ją zdefiniować z poziomu juli wywołując `ENV["TEST_PERFORMANCE"] = true`

W celu uruchomienia testów należy wywołać komendę
```
] test
```
Parametry testów znajdują się w pliku `tests/runtests.jl`