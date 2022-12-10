import os

from mido import MidiFile


mid = MidiFile('./audio/midiFiles/minecraft.mid', clip=True)
songs = []

mid.print_tracks(meta_only=True)