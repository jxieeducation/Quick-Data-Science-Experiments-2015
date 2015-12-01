object P10{
	def compress(ls: List[Any]): List[Any] = {
		var out:List[List[Any]] = List()
		
		var latest:Any = None
		var latest_count:Int = 0

		for (elem <- ls){
			if (latest == None){
				latest = elem
				latest_count = 1
			} else if (latest == elem){
				latest_count += 1
			} else {
				val out2 = List(latest, latest_count) :: out
				out = out2.reverse
				latest = elem
				latest_count = 1
			}

			if (ls.indexOf(elem) == ls.length - 1){
				val out2 = List(latest, latest_count) :: out
				out = out2.reverse			
			}

		}

		return out
		
  	}

	def main(args: Array[String]) {
		val list = List('a', 'a', 'a', 'b', 'c')
    	println(compress(list))
	}
}
