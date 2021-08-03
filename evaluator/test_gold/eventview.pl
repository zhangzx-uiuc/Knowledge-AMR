#!/usr/bin/perl
require 5.000;
use strict;

my $gdir = '';			# gold directory, where gold .txt and .a1 files are.

my  $opt_string = 'g:oxh';
our %opt;

use Getopt::Std;
getopts("$opt_string", \%opt) or usage();
usage() if $opt{h};
usage() if $#ARGV < 0;
if ($opt{g}) {$gdir = $opt{g}; $gdir =~ s/\/$//}

## annotations.
#  - they are referenced globally.
#  - should be initialized for every file.
my ($text, $textlen);
my (%anno, %mod);

my $fstem; # $fstem is globally referenced to know what exact file is under processing.
foreach my $fname (@ARGV) {
    my $tdir = '';			# target directory
    if ($fname =~ /^(.*\/)?([^\/]+)\.a2$/) {$tdir = $1; $fstem = $2}
    else {warn "unrecognizable filename: $fname.\n"; next} 

    if (!$tdir) {$tdir = '.'}
    if (!$gdir) {$gdir = $tdir}

    ## local initialization of annotations
    $text = '';
    %anno = %mod = ();

    # read text file
    if (!open (FILE, "<:utf8", "$gdir/$fstem.txt")) {warn "cannot open the text file: $gdir/$fstem.txt\n"; next}
    while (<FILE>) {$text .= $_}
    close (FILE);
    $textlen = length $text;

    # read annotatin files
    &read_so_file ("$gdir/$fstem.a1") or next;
    &read_so_file ($fname) or next;

    # output
    my @eid = ();
    foreach (keys %anno) {if ($_ =~ /^(E|R)/) {push @eid, $_}}
    @eid = sort {${$anno{${$anno{$a}}[1]}}[1] <=> ${$anno{${$anno{$b}}[1]}}[1]} @eid;

    foreach (@eid) {
	my ($type, $etid, @arg) = @{$anno{$_}};
	$type =~ s/^Positive_r/+R/;
	$type =~ s/^Negative_r/-R/;


	my $mod = $mod{$_}? $mod{$_} : ' ';
	my $exp = "$fstem-$_\t$mod $type\t";

	my @range = ();

	if ($etid) {
	    $exp .= '<Exp>' . &tspan($etid) . ' ';
	    push @range, [&trange($etid)];
	} # if

	foreach (@arg) {
	    my ($atype, $atid) = split ':', $_;
	    $exp .= "<$atype>" . &tspan($atid) . ' ';

	    if (&trange($atid)) {push @range, [&trange($atid)]}
	} # foreach

	$exp =~ s/ $//;

	@range = sort {$$a[0] <=> $$b[0]} @range;
	@range = &range_uniq (@range);
	my $excerpt = substr ($text, $range[0][0], $range[$#range][1] - $range[0][0]);

	my $base = $range[0][0];
	my @erange = map {[$$_[0] - $base, $$_[1] - $base]} @range;

	for (my $i = $#erange; $i >= 0; $i--) {
	    substr ($excerpt, ${$erange[$i]}[1], 0) = ']';
	    substr ($excerpt, ${$erange[$i]}[0], 0) = '[';
	} # for

	if ($opt{x}) {$exp .= "\t<Excerpt>$excerpt"}

	print "$exp\n";
    } # foreach
} # foreach


sub tspan {
    my ($id) = @_;
    if ($id =~ /^[TV]/) {
	my ($beg, $end) = (${$anno{$id}}[1], ${$anno{$id}}[2]);
	if ($opt{o}) {return substr ($text, $beg, $end - $beg) . "[$beg-$end]"}
	else         {return substr ($text, $beg, $end - $beg)}
    } # if
    else {return $id}
} # tspan


sub trange {
    my ($id) = @_;
    if ($id =~ /^[TV]/) {return (${$anno{$id}}[1], ${$anno{$id}}[2])}
} # tspan


sub range_uniq {
    my (@range) = @_;
    my %seen = ();
    my @urange = grep !$seen{"$$_[0]-$$_[0]"}++, @range;
    return @urange;
} # range_uniq


sub read_so_file {
    my ($fname) = @_;

    if (!open (FILE, "<", $fname)) {warn "cannot open the file: $fname\n"; return ''}
    my @line = <FILE>; chomp (@line);
    close (FILE);

    foreach (@line) {
	my ($id, $exp) = split /\t/;

	if (/^[TV]/) {
	    my ($type, $beg, $end) = split ' ', $exp;
	    $anno{$id} = [$type, $beg, $end];
	} # if

	elsif (/^(E|R)/) {
	    my @arg = split ' ', $exp;
	    my ($type, $tid) = split ':', shift @arg;

	    if ($tid) {
		if (($tid !~ /^[TV][0-9]+$/) || (!$anno{$tid})) {warn "invalid reference to text annnotation: [$fstem]\t$_\n"}
		if ($type ne ${$anno{$tid}}[0]) {warn "inconsistent event/relation type: [$fstem-$id]\t$type : ${$anno{$tid}}[0]\n"}
	    } # if

	    $anno{$id} = [$type, $tid, @arg];
	} # elsif

	elsif (/^M/) {
	    my ($mod, $eid) = split ' ', $exp;
	    $mod = substr ($mod, 0, 1);

	    if (($mod ne 'N') && ($mod ne 'S')) {warn "invalid type of event modification: [$fstem]\t$_\n"}
	    if (($eid !~ /^(E|R)[0-9]+$/) || !$anno{$eid}) {warn "invalid reference to event/relation annotation: [$fstem]\t$_\n"}

	    if ($id =~ /^M/) {
		if ($mod{$eid} && ($mod{$eid} ne $mod)) {$mod{$eid} = 'Z'}
		else                                    {$mod{$eid} = $mod}
	    } # if
	} # elsif

	elsif (/^*/) {}

	else {warn "invalid ID prefix: [$fstem]\t$_\n"}
    } # foreach

    return 1;
} # read_so_file


sub usage() {
    warn << "EOF";

[eventview]
last updated by jdkim\@dbcls.rois.ac.jp, 13 February, 2011.


<DESCRIPTION>
It shows in a human-readable format the standoff annotation of
the GENIA event extraction task of BioNLP Shared Task 2011.

It is designed to be conveniently used together with standard
unix tools, e.g. grep and cut.


<OUTPUT>
DESCRIPTION : PMID-event_ID [TAB] [N|S|Z] event_class [TAB] <Exp>event_trigger <ARG_type>ARG_text ...
EXAMPLE     : 9796702-E9                N +Regulation       <Exp>induce <Theme>Akt <Cause>p21ras
* 'N' stands for 'Negation', 'S' for 'Speculation', and 'Z' for both.


<USAGE>
$0 [-$opt_string] a2_file(s)


<OPTIONS>
-g gold_dir  specifies the directory where the *.txt and *.a1 files are.
             It defaults to the the same directory of the input *.a2 files.
-o           tells it to show the offsets of each text span.
-x           tells it to show the excerpts of each event.
-h           this (help) message.


<REFERENCE>
https://sites.google.com/site/bionlpst/

EOF
    exit;
}
