package com.neuronet

class MultiplyGate(u0c:Unit_, u1c:Unit_){
	var u0: Unit_ = u0c
	var u1: Unit_ = u1c
	var utop: Unit_ = new Unit_(0, 0)

	def forward(): Unit_ = {
		utop.value = u0.value * u1.value
		utop.gradient = 0.0
		return utop
	}

	def backprop() {
		u0.gradient += u1.value * utop.gradient
		u1.gradient += u0.value * utop.gradient
	}

	override def toString(): String = "u0: " + u0 + ", u1:" +  u1 + ", utop:" + utop;
}

