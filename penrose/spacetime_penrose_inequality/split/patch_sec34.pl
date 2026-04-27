#!/usr/bin/env perl
use strict;
use warnings;

my $file = "sec_34_logical_structure_and_gap_closure.tex";
open(my $fh, '<', $file) or die "Cannot open $file: $!";
my $content = do { local $/; <$fh> };
close($fh);

$content =~ s/upgrade to pointwise.*?general \$k\)\./upgrade to pointwise \$\tr_\Sigma k \ge 0\$ is exactly Theorem~C; this was previously a major gap, but Theorem~\ref\{thm:GapClosed\} below now provides the full resolution via the spectral deformation method, even for non-self-adjoint stability operators./s;

$content =~ s/Conclusion:\} The favorable condition/Conclusion:\} Theorem~\ref\{thm:GapClosed\} ensures that the favorable jump condition \$\tr_\Sigma k \ge 0\$/s;

open($fh, '>', $file) or die "Cannot write $file: $!";
print $fh $content;
close($fh);
