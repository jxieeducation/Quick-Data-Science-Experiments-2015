object P17{
	def split(n:Int, ls: List[Any]): (List[Any], List[Any]) = (n, ls) match {
		case (_, Nil) => (Nil, Nil)
		case (0, list) => (Nil, list)
		case (n, h::tail) => {
			val (pre, post) = split(n-1, tail)
			return (h::pre, post)
		}
  	}

	def main(args: Array[String]) {
		val list = List('a', 'a', 'a', 'b', 'c')
    	println(split(2, list))
	}
}
