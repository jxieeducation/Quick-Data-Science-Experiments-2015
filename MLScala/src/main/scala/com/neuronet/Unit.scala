package com.neuronet

class Unit_(vc:Double, gc: Double){
	var value: Double = vc;
	var gradient: Double = gc;
	override def toString(): String = "val: " + value + ", grad:" +  gradient;
}
