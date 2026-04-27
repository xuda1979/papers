#!/bin/bash
printf "\ProvidesPackage{cleveref}[2023/04/10 dummy cleveref]
" > cleveref.sty
printf "\DeclareOption*{}\ProcessOptions
" >> cleveref.sty
printf "\RequirePackage{hyperref}
" >> cleveref.sty
printf "\newcommand{\cref}[1]{\ref{#1}}
" >> cleveref.sty
printf "\newcommand{\Cref}[1]{\ref{#1}}
" >> cleveref.sty
printf "\newcommand{\cpageref}[1]{\pageref{#1}}
" >> cleveref.sty
printf "\newcommand{\Cpageref}[1]{\pageref{#1}}
" >> cleveref.sty
printf "\endinput
" >> cleveref.sty

printf "\ProvidesPackage{framed}[2023/04/10 dummy framed]
" > framed.sty
printf "\newenvironment{framed}{}{}
" >> framed.sty
printf "\newenvironment{oframed}{}{}
" >> framed.sty
printf "\endinput
" >> framed.sty
