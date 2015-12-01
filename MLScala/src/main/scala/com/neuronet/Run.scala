package com.neuronet {
  object RunGate extends App {
    var u0: Unit_ = new Unit_(1.0, 1.0)
    var u1: Unit_ = new Unit_(3.0, 1.0)
    var n: MultiplyGate = new MultiplyGate(u0, u1)
    var utop: Unit_ = n.forward()
    println(n)
    utop.gradient = 3.0
    n.backprop()
    println(n)
  }
}
