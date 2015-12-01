object P07{
	def flatten(ls: List[Any]): List[Any] = ls flatMap {
    	case ms: List[_] => flatten(ms)
    	case e => List(e)
  	}

	def main(args: Array[String]) {
		val list = List(List(1, 1), 2, List(3, List(5, 8)))
    	println(flatten(list))
	}
}
